"""
Part B - Sales Question Answering
=================================

Answers plain-English questions about ``sales.csv`` such as

    "What were the top 3 dishes yesterday?"
    "How much did we sell today?"
    "How many butter naan were sold this week?"

Approach: rule-based parsing  ->  structured ``Query``  ->  pandas does the maths.
-------------------------------------------------------------------------------
    question --parse--> Query(intent, metric, period, dishes/category, n, ...) --execute--> Answer

* The parser only decides WHAT to compute. All numbers come from pandas, never from guessing.
* The ``Query`` object is exactly the JSON an LLM could be asked to produce; swapping the
  rule-based parser for an LLM call would not change the executor at all.
* Anything the parser is unsure about ends in a polite message, never a made-up number:
  unknown dishes, dates outside the data, questions the CSV cannot answer (profit, customers ...).

Run ``python part_b_sales_qa.py --help`` for the CLI (interactive loop, --ask, --demo, --charts).
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CURRENCY = "\u20b9"  # rupee sign; sales.csv has no currency column, INR is an assumption

EXAMPLE_QUESTIONS = [
    "What were the top 3 dishes yesterday?",
    "How much did we sell today?",
    "How many butter naan were sold this week?",
    "Which day had the highest sales?",
    "Show me sales by category last week",
    "How many panner butter masla did we sell this month?",
    "Top 3 desserts in the last 7 days",
    "What is the capital of France?",
]


# =========================================================================== #
# Small text helpers
# =========================================================================== #
def _stem(tok: str) -> str:
    """Very light plural stripping: naans -> naan, desserts -> dessert (applied to both sides)."""
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _words(text: str) -> list[str]:
    """lower-case alphanumeric words; apostrophes dropped ("what's" -> "whats")."""
    text = str(text).lower().replace("&", " and ").replace("'", "").replace("\u2019", "")
    return re.findall(r"[a-z0-9]+", text)


def _tokens(text: str) -> list[str]:
    """``_words`` with plurals stripped, so "naans" matches "naan"."""
    return [_stem(w) for w in _words(text)]


def _key(text: str) -> str:
    return " ".join(_tokens(text))


def fmt_int(n: int) -> str:
    """Indian digit grouping for counts: 2255 -> 2,255; 123456 -> 1,23,456."""
    return fmt_money(n).replace(CURRENCY, "").replace(".00", "")


def fmt_money(x: float) -> str:
    """Indian digit grouping: 309530 -> ₹3,09,530. Decimals only if needed."""
    x = float(x)
    sign = "-" if x < 0 else ""
    whole, frac = f"{abs(x):.2f}".split(".")
    head, tail = whole[:-3], whole[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    whole = f"{head},{tail}" if head else tail
    return f"{sign}{CURRENCY}{whole}" + ("" if frac == "00" else f".{frac}")


def fmt_day(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def _fmt_range(a: date, b: date) -> str:
    if a == b:
        return fmt_day(a)
    left = f"{a:%a} {a.day} {a:%b}" if a.year == b.year else fmt_day(a)
    return f"{left} \u2013 {fmt_day(b)}"


def _plural(n: int, word: str) -> str:
    """_plural(1, 'dish') -> '1 dish'; _plural(3, 'dish') -> '3 dishes'."""
    if n == 1:
        return f"{n} {word}"
    es = word.endswith(("s", "sh", "ch", "x"))
    return f"{fmt_int(n)} {word}{'es' if es else 's'}"


# =========================================================================== #
# Data loading and cleaning
# =========================================================================== #
def clean_sales(raw: pd.DataFrame, dishes: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate and tidy sales rows. Returns ``(df, report)``.

    Adds ``revenue = quantity * price`` (price is a per-unit price: it is constant per item in the
    sample file), a python-``date`` column ``day`` and the dish ``category`` (from dishes.csv).
    Exact duplicate rows are *kept* (two identical lines can be two genuine orders) but counted.
    """
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"date", "item", "quantity", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"sales data is missing column(s): {sorted(missing)}")
    report = {"rows_in": len(df)}

    df["item"] = df["item"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    bad = (df[["date", "item", "quantity", "price"]].isna().any(axis=1)
           | (df["item"] == "") | (df["quantity"] <= 0) | (df["price"] < 0))
    report["invalid_rows_dropped"] = int(bad.sum())
    df = df[~bad].copy()
    report["exact_duplicate_rows_kept"] = int(df.duplicated(["date", "item", "quantity", "price"]).sum())

    df["quantity"] = df["quantity"].astype(int)
    df["date"] = df["date"].dt.normalize()
    df["day"] = df["date"].dt.date
    df["revenue"] = df["quantity"] * df["price"]

    # One spelling per dish: prefer the one in dishes.csv, else the first one seen.
    canon: dict[str, str] = {}
    if dishes is not None:
        for name in dishes["dish_name"].astype(str).str.strip():
            canon.setdefault(_key(name), re.sub(r"\s+", " ", name))
    for name in df["item"]:
        canon.setdefault(_key(name), str(name))
    df["item"] = df["item"].map(lambda n: canon[_key(n)])

    cat_by_key = {}
    if dishes is not None:
        cat_by_key = {_key(n): c for n, c in zip(dishes["dish_name"], dishes["category"])}
    df["category"] = df["item"].map(lambda n: cat_by_key.get(_key(n), "Uncategorised"))

    report["rows_out"] = len(df)
    return df.sort_values(["date", "item"]).reset_index(drop=True), report


def load_data(sales_path=None, dishes_path=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dishes = pd.read_csv(dishes_path or DATA_DIR / "dishes.csv")
    sales, report = clean_sales(pd.read_csv(sales_path or DATA_DIR / "sales.csv"), dishes)
    return sales, dishes, report


# =========================================================================== #
# Structured objects
# =========================================================================== #
@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str          # "yesterday (Mon 16 Nov 2026)"
    implicit: bool = False  # True when the user gave no period and we fell back to all data


@dataclass
class Query:
    """What the user wants - the structured form of a question."""
    intent: str                      # total | rank_items | rank_categories | breakdown_category |
                                     # breakdown_day | best_day | daily_average | price
    period: Period
    metric: str = "revenue"          # revenue | units | both
    n: int = 1
    ascending: bool = False          # True for "least sold"
    dishes: list[str] = field(default_factory=list)
    category: str | None = None
    group_note: str | None = None    # e.g. 'dishes with "naan" in the name'
    corrections: list[tuple[str, str]] = field(default_factory=list)  # misspelling fixes made

    def describe(self) -> str:
        what = self.dishes if len(self.dishes) <= 3 else f"{len(self.dishes)} dishes"
        return (f"intent={self.intent} metric={self.metric} n={self.n} dishes={what} "
                f"category={self.category} period={self.period.start}..{self.period.end}")


@dataclass
class Answer:
    text: str
    ok: bool = True
    kind: str = "answer"                  # answer | fallback | unsupported | not_found | no_data
    query: Query | None = None
    table: pd.DataFrame | None = None

    def __str__(self) -> str:
        return self.text


class ParseError(Exception):
    """The question can't be turned into a Query; ``answer`` is what we tell the user."""
    def __init__(self, text: str, kind: str = "fallback"):
        super().__init__(text)
        self.text, self.kind = text, kind


# =========================================================================== #
# Period parsing ("yesterday", "this week", "5 nov", "last 7 days" ...)
# =========================================================================== #
_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
           "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
           "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
           "december": 12, "dec": 12}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}
_NUM = r"\d+|" + "|".join(_NUMWORDS)


def _to_int(s: str) -> int:
    return int(s) if s.isdigit() else _NUMWORDS[s]


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - timedelta(days=1)


def _make_date(y: int, m: int, d: int) -> date:
    try:
        return date(y, m, d)
    except ValueError:
        raise ParseError(f"I couldn't read that as a valid date ({d}/{m}/{y}). "
                         "Try something like '5 November' or '2026-11-05'.")


def parse_period(text: str, as_of: date, data_start: date) -> tuple[Period, str]:
    """Find the single time period in ``text``. Returns ``(period, text_without_the_period)``.

    No period mentioned -> all available data (flagged ``implicit``).
    Weeks start on Monday. "last N days" includes today (a rolling window).
    """
    work = text.lower()
    found: list[Period] = []

    def take(pattern: str, build) -> None:
        nonlocal work
        for m in list(re.finditer(pattern, work)):
            period = build(m)
            if period is not None:
                found.append(period)
        work = re.sub(pattern, " ", work)  # blank it so shorter patterns can't re-match inside it

    def day(d: date, prefix: str) -> Period:
        return Period(d, d, f"{prefix} ({fmt_day(d)})".strip() if prefix else f"on {fmt_day(d)}")

    def rng(a: date, b: date, name: str) -> Period:
        return Period(a, b, f"{name} ({_fmt_range(a, b)})")

    monday = as_of - timedelta(days=as_of.weekday())
    first_this, _ = _month_bounds(as_of.year, as_of.month)
    last_prev = first_this - timedelta(days=1)

    # --- relative expressions (longest first) ---
    take(r"\bday before yesterday\b", lambda m: day(as_of - timedelta(days=2), "the day before yesterday"))
    take(rf"\b(?:in |over |during |for )?(?:the )?(?:last|past|previous)\s+({_NUM})\s+(day|week)s?\b",
         lambda m: rng(as_of - timedelta(days=_to_int(m[1]) * (7 if m[2] == "week" else 1) - 1), as_of,
                       f"in the last {_to_int(m[1])} {m[2]}{'' if _to_int(m[1]) == 1 else 's'}"))
    take(r"\b(?:last|previous)\s+week\b",
         lambda m: rng(monday - timedelta(days=7), monday - timedelta(days=1), "last week"))
    take(r"\b(?:this|current)\s+week\b|\bweek to date\b",
         lambda m: rng(monday, as_of, "this week"))
    take(r"\b(?:last|previous)\s+month\b",
         lambda m: rng(*_month_bounds(last_prev.year, last_prev.month), "last month"))
    take(r"\b(?:this|current)\s+month\b|\bmonth to date\b",
         lambda m: rng(first_this, as_of, "this month"))
    take(r"\b(?:this year|year to date)\b", lambda m: rng(date(as_of.year, 1, 1), as_of, "this year"))
    take(r"\byesterday\b", lambda m: day(as_of - timedelta(days=1), "yesterday"))
    take(r"\btoday\b|\btonight\b|\bright now\b", lambda m: day(as_of, "today"))

    # --- explicit dates ---
    def explicit(y: int | None, mth: int, d: int) -> Period:
        if y is None:
            y = as_of.year
            if _make_date(y, mth, d) > as_of:
                y -= 1
        return day(_make_date(y, mth, d), "")

    take(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: explicit(int(m[1]), int(m[2]), int(m[3])))
    take(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b", lambda m: explicit(int(m[3]), int(m[2]), int(m[1])))  # dd/mm/yyyy
    take(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})\b(?:\s*,?\s*(\d{{4}})\b)?",
         lambda m: explicit(int(m[3]) if m[3] else None, _MONTHS[m[2]], int(m[1])))
    take(rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{{4}})\b)?",
         lambda m: explicit(int(m[3]) if m[3] else None, _MONTHS[m[1]], int(m[2])))
    take(rf"\b({_MON})\s+(\d{{4}})\b", lambda m: rng(*_month_bounds(int(m[2]), _MONTHS[m[1]]),
                                                    f"in {m[1].title()} {m[2]}"))

    # --- weekdays and whole named months ---
    def weekday(m) -> Period:
        idx = _WEEKDAYS.index(m[2])
        if m[1] == "last":                                   # same weekday in the previous week
            d = monday - timedelta(days=7) + timedelta(days=idx)
        else:                                                # most recent such weekday up to today
            d = as_of - timedelta(days=(as_of.weekday() - idx) % 7)
        return day(d, "")

    take(rf"\b(?:(last|this|on)\s+)?({'|'.join(_WEEKDAYS)})\b", weekday)

    def named_month(m) -> Period:
        mth = _MONTHS[m[1]]
        y = as_of.year if mth <= as_of.month else as_of.year - 1
        return rng(*_month_bounds(y, mth), f"in {date(y, mth, 1):%B %Y}")

    take(rf"\b(?:in|during|for|of)\s+({_MON})\b", named_month)

    # --- all time ---
    all_time = [None]
    take(r"\ball[- ]time\b|\boverall\b|\bever\b|\btill date\b|\btill now\b|\bto date\b|\bsince the beginning\b"
         r"|\bin total\b|\baltogether\b|\blifetime\b|\bentire\b",
         lambda m: all_time.__setitem__(0, Period(data_start, as_of, f"across all available data "
                                                  f"({_fmt_range(data_start, as_of)})")))
    if all_time[0]:
        found.append(all_time[0])

    if len(found) > 1:
        raise ParseError("I can only answer for one time period at a time (for example 'this week' "
                         "or 'yesterday'). Could you ask about each period separately?")
    if found:
        return found[0], work
    return Period(data_start, as_of, f"across all available data ({_fmt_range(data_start, as_of)})",
                  implicit=True), work


# =========================================================================== #
# Dish / category resolution (exact -> spelling-corrected -> partial -> category)
# =========================================================================== #
_STOP_WORDS = """
a an the of for in on at by to from and with or is are was were be been being do does did done have has had
what whats which who whom whose when where how why that this these those it its there here
we us our you your i me my mine they their he she him her
many much total number count amount
sell sells sold selling sale sales soldout order orders ordered ordering
dish dishes item items food menu product products unit units quantity qty plate plates portion portions
piece pieces serving servings
top best bottom worst least most highest lowest popular bestseller bestselling seller sellers favourite favorite
revenue earn earned earning earnings income money rupee rupees rs inr worth turnover collection collected
make made making get got give show tell list display find see check report
please pls kindly hey hello hi thanks thank
day days daily week weeks month months year years date dates today yesterday tonight
so far till now ever overall far yet
outlet restaurant kharchify business shop store cafe
can could would should will shall may might
any all some per each every
category categories wise
average avg mean busiest quietest peak slowest fewest weakest price priced pricing cost costs rate
perform performing performer performers performance
than then also just only about around approximately roughly
""".split()
_STOP = {_stem(w) for w in _STOP_WORDS}

_CATEGORY_ALIASES = {
    ("starter",): "Starter", ("appetizer",): "Starter",
    ("main", "course"): "Main Course", ("main",): "Main Course",
    ("bread",): "Bread", ("rice",): "Rice", ("dessert",): "Dessert",
    ("beverage",): "Beverage", ("drink",): "Beverage",
}


@dataclass
class Resolution:
    """Which dish(es)/category a question mentions. All-empty means 'no dish mentioned'."""
    dishes: list[str] = field(default_factory=list)
    category: str | None = None
    note: str | None = None                       # how a group was matched
    corrections: list[tuple[str, str]] = field(default_factory=list)
    unresolved: str | None = None                 # something dish-like we could NOT find
    hint: str = ""                                # "Did you mean ...?"
    exact: bool = False                           # every dish named in full, no guessing involved

    @property
    def found(self) -> bool:
        return bool(self.dishes or self.category)


class Catalog:
    """Every dish name we know (dishes.csv plus anything seen in sales)."""

    def __init__(self, names: list[str]):
        self.names = sorted(set(names))
        self.tok = {n: _tokens(n) for n in self.names}
        self.vocab = {t for toks in self.tok.values() for t in toks}

    def _exact_spans(self, toks: list[str]) -> list[tuple[int, int, str]]:
        """Dish names that appear as a whole word-sequence inside ``toks`` (longest, non-overlapping)."""
        hits = []
        for name, nt in self.tok.items():
            L = len(nt)
            for i in range(len(toks) - L + 1):
                if toks[i:i + L] == nt:
                    hits.append((i, i + L, name))
        hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        chosen, used = [], set()
        for s, e, name in hits:
            if used.isdisjoint(range(s, e)):
                chosen.append((s, e, name))
                used.update(range(s, e))
        return sorted(chosen)

    def _correct(self, tok: str) -> str:
        """Fix a probable typo against known dish words ("panner" -> "paneer")."""
        if tok in self.vocab or tok in _STOP or len(tok) < 4 or tok.isdigit():
            return tok
        # similar length as well as similar letters ("weather" must not become "water")
        close = [c for c in difflib.get_close_matches(tok, self.vocab, n=3, cutoff=0.8)
                 if abs(len(c) - len(tok)) <= (1 if len(tok) < 8 else 2)]
        return close[0] if close else tok

    def resolve(self, toks: list[str], ignore: set[int], words: list[str] | None = None) -> Resolution:
        """Find the dish(es)/category mentioned in ``toks`` (``ignore`` = positions to skip).

        Order: exact dish name -> exact after typo-fixing -> category word -> every dish containing
        the word(s) ("naan" -> all naans) -> same with typo-fixing. Never raises; anything dish-like
        that can't be matched is reported in ``unresolved`` (with a "did you mean" hint).
        """
        words = words or toks

        def leftover_idx(tokens, spans):
            used = {i for s, e, _ in spans for i in range(s, e)} | ignore
            return [i for i, t in enumerate(tokens) if i not in used and t not in _STOP]

        def leftover(tokens, spans):
            return [tokens[i] for i in leftover_idx(tokens, spans)]

        spans = self._exact_spans(toks)
        left_idx = leftover_idx(toks, spans)
        left = [toks[i] for i in left_idx]
        shown = " ".join(words[i] for i in left_idx)      # what the user actually typed
        if spans and not left:
            return Resolution([n for _, _, n in spans], exact=True)
        if not spans and not left:
            return Resolution()

        if not spans:
            # typo-fix each word, then try exact names again ("panner butter masla")
            fixed = [t if i in ignore else self._correct(t) for i, t in enumerate(toks)]
            if fixed != toks:
                spans2 = self._exact_spans(fixed)
                if spans2 and not leftover(fixed, spans2):
                    orig = " ".join(words[i] for i, t in enumerate(toks) if i not in ignore and t not in _STOP)
                    names = [n for _, _, n in spans2]
                    return Resolution(names, corrections=[(orig, ", ".join(names))])
            cat = _CATEGORY_ALIASES.get(tuple(left))
            if cat:
                return Resolution(category=cat, note=f'the "{cat}" category')
            phrase = " ".join(left)
            partial = [n for n, nt in self.tok.items() if set(left) <= set(nt)]
            if partial:
                return Resolution(partial, note=f'dishes with "{phrase}" in the name')
            fixed_left = [self._correct(t) for t in left]
            partial = [n for n, nt in self.tok.items() if set(fixed_left) <= set(nt)]
            if partial and fixed_left != left:
                return Resolution(partial, note=f'dishes with "{" ".join(fixed_left)}" in the name',
                                  corrections=[(shown, " ".join(fixed_left))])

        phrase = shown
        by_key = {" ".join(t): n for n, t in self.tok.items()}
        close = difflib.get_close_matches(phrase, list(by_key), n=2, cutoff=0.6)
        hint = f" Did you mean {' or '.join(by_key[c] for c in close)}?" if close else ""
        return Resolution(unresolved=phrase, hint=hint)


# =========================================================================== #
# Question -> Query
# =========================================================================== #
_UNSUPPORTED = [
    (r"\b(order value|aov|average (?:bill|ticket|basket)|per order|per bill|bill value)\b",
     "sales.csv has no order or bill IDs, so I can't work out per-order figures such as average order value."),
    (r"\b(?:on|in|from) (?:the |our )?menu\b|\bmenu (?:items|size|list)\b",
     "That's a question about the menu rather than sales; I can only answer from the sales records."),
    (r"\b(profit|margin|expenses?|cogs|loss(?:es)?|food cost)\b",
     "sales.csv only has quantities and selling prices, so I can't work out profit, margin or costs. "
     "I can tell you revenue, though."),
    (r"\b(stock|inventory|ingredients?|restock|supplier|wastage|waste)\b",
     "That needs inventory data, which isn't in sales.csv. I can only answer from sales records."),
    (r"\b(customers?|guests?|tables?|waiters?|staff|employees?|cashiers?|footfall|people|visitors?|covers)\b",
     "sales.csv has no customer, table or staff information, so I can't answer that."),
    (r"\b(discounts?|tax(?:es)?|gst|tips?|refunds?|cancell?(?:ed|ations?)|payments?|upi|cash)\b",
     "sales.csv has no discount, tax, payment or refund details, so I can't answer that."),
    (r"\b(hours?|hourly|morning|afternoon|evening|lunch|dinner|breakfast|peak time|rush)\b",
     "sales.csv records dates only (no time of day), so I can't break sales down by hour or meal."),
    (r"\b(forecast|predict\w*|projection|projected|expected|upcoming|tomorrow|next (?:week|month|year|day)|"
     r"will we|are we going to|going to sell)\b",
     "I can only report on sales that have already happened, not forecast future ones."),
    (r"\bwhy\b",
     "I can report what the numbers are, but I can't explain why they changed. "
     "Try asking for the numbers (for example sales by day) and compare them."),
]
_SALES_WORDS = {_stem(w) for w in
                "sell sold selling sale sales revenue earn earned earning income turnover dish dishes item items "
                "order orders unit units quantity popular bestseller bestselling top best least worst money "
                "rupee rupees collection collected made make".split()}


_STRONG_CUES = {_stem(w) for w in
                "sell sold selling sale sales revenue earn earned earning income order orders unit units "
                "quantity dish dishes item items".split()}


def _find_n(text: str) -> tuple[int | None, str]:
    """"top 3", "five best", "bottom 2" -> (3, text_without_that_phrase)."""
    pats = [rf"\b(?:top|bottom|best|worst|least|first|lowest|highest)\s+({_NUM})\b",
            rf"\b({_NUM})\s+(?:best|top|most|least|worst|bottom|highest|lowest)\b"]
    for p in pats:
        m = re.search(p, text)
        if m:
            return _to_int(m[1]), text[:m.start()] + " " + text[m.end():]
    return None, text


_HAS = lambda p, t: re.search(p, t) is not None  # noqa: E731

_TOP = (r"\b(top|best|bestsell\w*|best sell\w*|popular|favou?rite|most|highest sell\w*|hot sell\w*|"
        r"sold the most|sells? the most)\b")
_BOTTOM = r"\b(bottom|worst|least|lowest|slowest|fewest|weakest|slow moving)\b"
_UNITS = (r"\bhow many\b|\bunits?\b|\bquantit\w*\b|\bqty\b|\bnumber of\b|\bplates?\b|\bportions?\b|"
          r"\bpieces?\b|\bservings?\b|\bcount\b|\bpcs\b")
_MONEY = (r"\brevenue\b|\bearn\w*\b|\bincome\b|\bmoney\b|\brupees?\b|\brs\b|\binr\b|\bworth\b|\bturnover\b|"
          r"\bcollect\w*\b|\bamount\b|\bvalue\b|\bgross\b|\u20b9")


class SalesAssistant:
    """Parses a question, runs it on the data with pandas, and phrases the answer."""

    def __init__(self, sales: pd.DataFrame, dishes: pd.DataFrame | None = None, as_of: date | None = None):
        if sales.empty:
            raise ValueError("sales data is empty")
        self.sales = sales
        self.data_start, self.data_end = sales["day"].min(), sales["day"].max()
        # "today" defaults to the last date in the file so the sample data always answers
        # reproducibly; pass as_of=date.today() for a live system.
        self.as_of = as_of or self.data_end
        names = list(sales["item"].unique())
        if dishes is not None:
            names += [re.sub(r"\s+", " ", str(n)).strip() for n in dishes["dish_name"]]
        self.catalog = Catalog(names)

    # ------------------------------------------------------------------ public
    def ask(self, question) -> Answer:
        """Never raises: unclear or unanswerable questions get a polite message."""
        try:
            query = self.parse(question)
            return self.execute(query)
        except ParseError as e:
            return Answer(e.text, ok=False, kind=e.kind)
        except Exception:  # last-resort safety net - never crash, never invent an answer
            return Answer(self._fallback_text("Something went wrong while reading that question."),
                          ok=False, kind="fallback")

    # ------------------------------------------------------------------ parsing
    def _fallback_text(self, lead: str = "Sorry, I'm not sure how to answer that.") -> str:
        ex = "\n".join(f"  - {q}" for q in EXAMPLE_QUESTIONS[:5])
        return (f"{lead} I'd rather not guess. I can answer questions about the sales data "
                f"({_fmt_range(self.data_start, self.data_end)}), for example:\n{ex}")

    def parse(self, question) -> Query:
        """Turn a question into a ``Query`` (or raise ParseError with a polite message)."""
        if not isinstance(question, str) or not question.strip():
            raise ParseError(self._fallback_text("I didn't catch a question."))
        text = re.sub(r"\s+", " ", question.lower()).strip()[:500]
        text = re.sub(r"(?<=\w)['\u2019]s\b", "", text)          # "yesterday's sales" -> "yesterday sales"

        # 1. things the CSV cannot answer get a specific, honest explanation
        for pattern, message in _UNSUPPORTED:
            if re.search(pattern, text):
                raise ParseError(message, kind="unsupported")

        # 2. time period (removed from the text so its words are not mistaken for dish names)
        period, rest = parse_period(text, self.as_of, self.data_start)
        if period.start > self.as_of:
            raise ParseError("I can only report on sales that have already happened.", kind="unsupported")

        # 3. cues that need the text *before* numbers are stripped out ("top 3" -> keep "top")
        wants_top = _HAS(_TOP, rest)
        ascending = _HAS(_BOTTOM, rest)
        wants_rank = wants_top or ascending
        plural = _HAS(r"\b(dishes|items|sellers|categories|days|products)\b", rest)
        has_units, has_money = _HAS(_UNITS, rest), _HAS(_MONEY, rest)
        how_much, how_many = _HAS(r"\bhow much\b", rest), _HAS(r"\bhow many\b", rest)
        per_day = _HAS(r"\b(per|a|each|every) day\b|\bdaily\b|\bday wise\b|\bday by day\b|\bby day\b", text)

        n, rest = _find_n(rest)
        if n is not None and n < 1:
            raise ParseError("Please ask for at least 1 item, for example 'top 3 dishes'.")

        # 4. which dish / category (if any)
        toks = _tokens(rest)
        res = self.catalog.resolve(toks, {i for i, t in enumerate(toks) if t.isdigit()}, _words(rest))
        sales_like = bool(_SALES_WORDS & set(_tokens(text)))

        strong = bool(_STRONG_CUES & set(_tokens(text)))
        cue = strong or how_many or how_much or wants_rank   # the question is clearly about sales

        if res.unresolved:   # a dish-like phrase we can't find
            if strong:
                raise ParseError(f'I couldn\'t find "{res.unresolved}" on the menu.{res.hint}', kind="not_found")
            raise ParseError(self._fallback_text())

        if res.found and not res.exact and not cue:   # guessed/partial match without a sales cue
            raise ParseError(self._fallback_text())

        def metric(default: str) -> str:
            if has_units and has_money:
                return "both"
            return "units" if has_units else "revenue" if has_money else default

        def mk(intent: str, **kw) -> Query:
            return Query(intent=intent, period=period, dishes=res.dishes, category=res.category,
                         group_note=res.note, corrections=res.corrections, **kw)

        # 5. which kind of question?  (order matters: most specific first)
        if _HAS(r"\border(s)?\b", rest) and not res.found and (how_many or _HAS(r"\bnumber of\b|\bcount\b", rest)):
            raise ParseError("sales.csv doesn't have order IDs or bill numbers, so I can't count orders. "
                             "I can tell you how many units were sold, or how much was sold, instead.",
                             kind="unsupported")

        if _HAS(r"\b(price|priced|pricing|cost|costs|rate)\b", rest):
            if res.dishes:
                return mk("price")
            raise ParseError("sales.csv doesn't have cost data. If you name a dish I can tell you its "
                             "selling price.", kind="unsupported")

        by_category = _HAS(r"\bcategor", text)
        day_word = _HAS(r"\b(day|days)\b", text)
        superlative_day = wants_rank or _HAS(r"\b(busiest|quietest|peak|highest|lowest|best|worst|slowest)\b", rest)

        if day_word and superlative_day and not per_day and not res.found:
            low = _HAS(r"\b(quietest|lowest|worst|slowest|least|fewest|bottom)\b", rest)
            return mk("best_day", n=n or (5 if plural else 1), ascending=low,
                      metric="units" if has_units and not has_money else "revenue")

        if _HAS(r"\b(average|avg|mean)\b", rest) or (per_day and (how_many or how_much)):
            return mk("daily_average", metric=metric("both"))

        if by_category and not res.found:
            if wants_rank:
                return mk("rank_categories", n=n or (6 if plural else 1), ascending=ascending,
                          metric=metric("revenue"))
            return mk("breakdown_category", metric=metric("revenue"))

        if per_day and not res.found:
            return mk("breakdown_day", metric=metric("revenue"))

        if wants_rank:
            if res.dishes:
                raise ParseError("I can rank all dishes, or the dishes in one category, but not the rank of "
                                 "one specific dish. Try 'top 5 dishes this week'.", kind="unsupported")
            return mk("rank_items", n=n or (5 if plural else 1), ascending=ascending, metric=metric("units"))

        if sales_like or res.found:
            if not res.found:
                default = "units" if (how_many and not how_much) else "revenue"
            elif how_much and not has_units and not has_money:
                default = "both"            # "how much biryani did we sell" is ambiguous -> show both
            else:
                default = "units" if how_many else "both"
            return mk("total", metric=metric(default))

        raise ParseError(self._fallback_text())

    # ------------------------------------------------------------------ execution
    def _slice(self, q: Query) -> pd.DataFrame:
        df = self.sales
        df = df[(df["day"] >= q.period.start) & (df["day"] <= q.period.end)]
        if q.dishes:
            df = df[df["item"].isin(q.dishes)]
        if q.category:
            df = df[df["category"] == q.category]
        return df

    def _coverage_note(self, q: Query) -> tuple[str | None, str | None]:
        """(blocking_message, warning). Blocks when the period has no data at all."""
        s, e = q.period.start, q.period.end
        if e < self.data_start or s > self.data_end:
            return (f"I don't have any sales data for {_fmt_range(s, e)}. "
                    f"The data covers {_fmt_range(self.data_start, self.data_end)}.", None)
        if s < self.data_start or e > self.data_end:
            return None, (f"Note: the data only covers {_fmt_range(self.data_start, self.data_end)}, "
                          f"so this figure is for part of that period only.")
        return None, None

    def execute(self, q: Query) -> Answer:
        block, warn = self._coverage_note(q)
        if block:
            return Answer(block, ok=False, kind="no_data", query=q)
        handler = getattr(self, f"_do_{q.intent}")
        ans: Answer = handler(q)
        ans.query = q
        extras = []
        for wrong, right in q.corrections:
            extras.append(f'Interpreted "{wrong}" as {right}.')
        if warn:
            extras.append(warn)
        span_days = (q.period.end - q.period.start).days + 1
        if q.period.label.startswith("this week") and span_days < 7:
            extras.append(f"(This week covers only {_plural(span_days, 'day')} so far; "
                          "ask for 'the last 7 days' for a rolling week.)")
        if extras:
            ans.text += "\n" + " ".join(extras)
        return ans

    # -- helpers -------------------------------------------------------------------------
    @staticmethod
    def _subject(q: Query) -> str:
        if q.category:
            return f"{q.category} dishes"
        if len(q.dishes) == 1:
            return q.dishes[0]
        if q.dishes:
            return f"{len(q.dishes)} {q.group_note}" if q.group_note else ", ".join(q.dishes)
        return "all dishes"

    @staticmethod
    def _agg(df: pd.DataFrame, by: str) -> pd.DataFrame:
        return (df.groupby(by, as_index=False)
                  .agg(units=("quantity", "sum"), revenue=("revenue", "sum")))

    # -- intents -------------------------------------------------------------------------
    def _do_total(self, q: Query) -> Answer:
        df = self._slice(q)
        units, revenue = int(df["quantity"].sum()), df["revenue"].sum()
        n_dishes = df["item"].nunique()
        subject = self._subject(q)
        whole_menu = subject == "all dishes"

        if whole_menu:
            if df.empty:
                return Answer(f"No sales are recorded for {q.period.label}.", kind="no_data")
            spread = f"{_plural(units, 'unit')} across {_plural(n_dishes, 'dish')}"
            if q.metric == "units":
                text = f"{_plural(units, 'unit')} sold {q.period.label}, across {_plural(n_dishes, 'different dish')} " \
                       f"(worth {fmt_money(revenue)})."
            else:
                text = f"Total sales {q.period.label}: {fmt_money(revenue)} ({spread})."
            return Answer(text)

        if q.metric == "revenue":
            text = f"{subject}: {fmt_money(revenue)} in sales {q.period.label} ({_plural(units, 'unit')})."
        else:
            text = f"{subject}: {_plural(units, 'unit')} sold {q.period.label}, worth {fmt_money(revenue)}."
        table = None
        if len(q.dishes) > 1:
            table = self._agg(df, "item").sort_values(["units", "item"], ascending=[False, True])
            lines = [f"  - {r.item}: {_plural(int(r.units), 'unit')} ({fmt_money(r.revenue)})" for r in table.itertuples()]
            lines += [f"  - {d}: 0 units" for d in q.dishes if d not in set(table["item"])]
            text += "\n" + "\n".join(lines[:12]) + ("\n  ..." if len(lines) > 12 else "")
        if df.empty and q.dishes and self.sales["item"].isin(q.dishes).sum() == 0:
            text += "\n(It's on the menu, but there are no sales recorded for it at all.)"
        return Answer(text, table=table)

    def _do_price(self, q: Query) -> Answer:
        prices = (self.sales[self.sales["item"].isin(q.dishes)]
                  .groupby("item")["price"].agg(["min", "max"]))
        lines = []
        for d in q.dishes:
            if d not in prices.index:
                lines.append(f"{d}: no sales yet, so I don't have its price.")
            elif prices.loc[d, "min"] == prices.loc[d, "max"]:
                lines.append(f"{d} is priced at {fmt_money(prices.loc[d, 'min'])}.")
            else:
                lines.append(f"{d} was sold at {fmt_money(prices.loc[d, 'min'])} to "
                             f"{fmt_money(prices.loc[d, 'max'])}.")
        return Answer("\n".join(lines[:10]))

    def _do_rank_items(self, q: Query) -> Answer:
        df = self._slice(q)
        if df.empty:
            return Answer(f"No sales are recorded for {q.period.label}.", kind="no_data")
        g = self._agg(df, "item")
        key, other = ("revenue", "units") if q.metric == "revenue" else ("units", "revenue")
        g = g.sort_values([key, other, "item"], ascending=[q.ascending, q.ascending, True]).reset_index(drop=True)
        top = g.head(q.n)
        cutoff = top[key].iloc[-1]
        tied = int(((g[key] == cutoff).sum()) - (top[key] == cutoff).sum())
        word = "Bottom" if q.ascending else "Top"
        unit_name = "revenue" if q.metric == "revenue" else "units sold"
        noun = f"{q.category.lower()} dish" if q.category else "dish"
        head = (f"{word} {noun} by {unit_name} {q.period.label}:" if len(top) == 1 else
                f"{word} {_plural(len(top), noun)} by {unit_name} {q.period.label}:")
        lines = [f"  {i}. {r.item} \u2014 {_plural(int(r.units), 'unit')} ({fmt_money(r.revenue)})"
                 for i, r in enumerate(top.itertuples(), 1)]
        text = head + "\n" + "\n".join(lines)
        if len(top) < q.n:
            text += f"\n(Only {len(g)} dish{'es' if len(g) != 1 else ''} sold in this period.)"
        if tied > 0:
            shown = _plural(int(cutoff), "unit") if key == "units" else fmt_money(cutoff)
            text += (f"\nNote: {_plural(tied, 'other dish') if tied == 1 else str(tied) + ' other dishes'} "
                     f"also {'sold' if key == 'units' else 'made'} {shown}, tied with the last place shown.")
        if q.ascending:
            text += "\n(Only dishes with at least one sale in this period are considered.)"
        return Answer(text, table=top)

    def _do_rank_categories(self, q: Query) -> Answer:
        g = self._agg(self._slice(q), "category")
        key = "revenue" if q.metric == "revenue" else "units"
        g = g.sort_values([key, "category"], ascending=[q.ascending, True]).head(q.n)
        lines = [f"  {i}. {r.category} \u2014 {_plural(int(r.units), 'unit')} ({fmt_money(r.revenue)})"
                 for i, r in enumerate(g.itertuples(), 1)]
        word = "lowest" if q.ascending else "highest"
        return Answer(f"Categories with the {word} {'revenue' if key == 'revenue' else 'units sold'} "
                      f"{q.period.label}:\n" + "\n".join(lines), table=g)

    def _do_breakdown_category(self, q: Query) -> Answer:
        g = self._agg(self._slice(q), "category")
        key = "revenue" if q.metric == "revenue" else "units"
        g = g.sort_values([key, "category"], ascending=[False, True])
        total_rev = g["revenue"].sum()
        if key == "units":
            lines = [f"  - {r.category}: {_plural(int(r.units), 'unit')} ({fmt_money(r.revenue)})" for r in g.itertuples()]
            total = f"\n  Total: {_plural(int(g['units'].sum()), 'unit')} ({fmt_money(total_rev)})"
        else:
            lines = [f"  - {r.category}: {fmt_money(r.revenue)} ({100 * r.revenue / total_rev:.0f}%), "
                     f"{_plural(int(r.units), 'unit')}" for r in g.itertuples()]
            total = f"\n  Total: {fmt_money(total_rev)}"
        return Answer(f"Sales by category {q.period.label}:\n" + "\n".join(lines) + total, table=g)

    def _do_breakdown_day(self, q: Query) -> Answer:
        g = self._agg(self._slice(q), "day")
        lines = [f"  - {fmt_day(r.day)}: {fmt_money(r.revenue)}, {_plural(int(r.units), 'unit')}" for r in g.itertuples()]
        if len(lines) > 14:
            lines = lines[:7] + [f"  ... ({len(lines) - 14} more days) ..."] + lines[-7:]
        return Answer(f"Sales by day {q.period.label}:\n" + "\n".join(lines), table=g)

    def _do_best_day(self, q: Query) -> Answer:
        g = self._agg(self._slice(q), "day")
        key = "units" if q.metric == "units" else "revenue"
        g = g.sort_values([key, "day"], ascending=[q.ascending, True]).head(q.n)
        word = "lowest" if q.ascending else "highest"
        lines = [f"  {i}. {fmt_day(r.day)} \u2014 {fmt_money(r.revenue)}, {_plural(int(r.units), 'unit')}"
                 for i, r in enumerate(g.itertuples(), 1)]
        if len(g) == 1:
            r = g.iloc[0]
            return Answer(f"The day with the {word} {'units sold' if key == 'units' else 'sales'} {q.period.label} "
                          f"was {fmt_day(r['day'])}: {fmt_money(r['revenue'])} from {int(r['units'])} units.", table=g)
        return Answer(f"Days with the {word} {'units sold' if key == 'units' else 'sales'} {q.period.label}:\n"
                      + "\n".join(lines), table=g)

    def _do_daily_average(self, q: Query) -> Answer:
        df = self._slice(q)
        s, e = max(q.period.start, self.data_start), min(q.period.end, self.data_end)
        days = (e - s).days + 1
        units, revenue = int(df["quantity"].sum()), df["revenue"].sum()
        subject = self._subject(q)
        scope = "" if subject == "all dishes" else f" for {subject}"
        text = (f"Average per day{scope} {q.period.label}, over {_plural(days, 'day')}: "
                f"{fmt_money(round(revenue / days))} in sales and {units / days:.1f} units.")
        return Answer(text)


# =========================================================================== #
# Charts (bonus)
# =========================================================================== #
def make_charts(sales: pd.DataFrame, out_dir: Path = OUT_DIR, show: bool = False) -> list[Path]:
    """Sales by category and by day. Returns the saved PNG paths."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    paths = []

    cat = sales.groupby("category")["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(cat.index, cat.values, color="#2a7ab0")
    ax.bar_label(bars, labels=[fmt_money(v) for v in cat.values], padding=3, fontsize=9)
    ax.set_xlim(0, cat.max() * 1.18)
    ax.set_title("Revenue by category (all data)")
    ax.set_xlabel(f"Revenue ({CURRENCY})")
    fig.tight_layout()
    paths.append(out_dir / "part_b_sales_by_category.png")
    fig.savefig(paths[-1], dpi=150)

    daily = sales.groupby("date")["revenue"].sum()
    weekend = daily.index.dayofweek >= 5
    fig2, ax2 = plt.subplots(figsize=(9, 3.8))
    ax2.bar(daily.index, daily.values, color=["#e08a1e" if w else "#2a7ab0" for w in weekend], width=0.8)
    ax2.set_title("Revenue by day (orange = weekend)")
    ax2.set_ylabel(f"Revenue ({CURRENCY})")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    paths.append(out_dir / "part_b_sales_by_day.png")
    fig2.savefig(paths[-1], dpi=150)
    if not show:
        plt.close("all")
    return paths


# =========================================================================== #
# CLI
# =========================================================================== #
def _build_assistant(as_of_arg: str | None) -> SalesAssistant:
    sales, dishes, _ = load_data()
    as_of = None
    if as_of_arg:
        as_of = date.today() if as_of_arg == "today" else date.fromisoformat(as_of_arg)
    return SalesAssistant(sales, dishes, as_of=as_of)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask plain-English questions about sales.csv")
    ap.add_argument("--ask", metavar="QUESTION", help="answer one question and exit")
    ap.add_argument("--demo", action="store_true", help="run the built-in example questions")
    ap.add_argument("--charts", action="store_true", help="save the sales charts to outputs/")
    ap.add_argument("--as-of", metavar="YYYY-MM-DD|today",
                    help="what 'today' means (default: the last date in sales.csv)")
    args = ap.parse_args(argv)
    bot = _build_assistant(args.as_of)
    print(f"[Data: {_fmt_range(bot.data_start, bot.data_end)} | 'today' = {fmt_day(bot.as_of)}]")

    if args.charts:
        for p in make_charts(bot.sales):
            print("saved", p)
    if args.ask:
        print(bot.ask(args.ask))
        return 0
    if args.demo:
        for q in EXAMPLE_QUESTIONS:
            print(f"\nQ: {q}\nA: {bot.ask(q)}")
        return 0
    if args.charts:
        return 0

    print("Ask a question about sales (type 'quit' to exit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"quit", "exit", "q"}:
            return 0
        if q:
            print(f"Assistant: {bot.ask(q)}")


if __name__ == "__main__":
    sys.exit(main())
