# Kharchify – AI/ML Intern Task

Two small AI features for a restaurant platform, in Python:

* **Part A** – suggest a menu category for a new dish (`part_a_classifier.py`)
* **Part B** – answer plain-English questions about sales (`part_b_sales_qa.py`)

The one-page write-up is in [`WRITEUP.md`](WRITEUP.md) (also `WRITEUP.pdf`). The notebook
[`Kharchify_Interview_Task.ipynb`](Kharchify_Interview_Task.ipynb) walks through both parts with all outputs already run.

## Quick start

```bash
pip install -r requirements.txt

python part_a_classifier.py                       # every Part A number: cleaning, CV, test set, errors, unseen dishes
python part_b_sales_qa.py --demo                  # example questions with answers
python part_b_sales_qa.py                          # interactive question loop (type 'quit' to exit)
python part_b_sales_qa.py --ask "How many butter naan were sold in the last 7 days?"
python part_b_sales_qa.py --charts                # saves charts to outputs/
python -m unittest discover -s tests -v           # 60 tests
```
Python 3.10+ (developed on 3.12). Everything is seeded (`SEED = 42`), so results are reproducible. No API keys are used or needed.

## Results

| | |
|---|---|
| Held-out test accuracy (stratified 80/20, seed 42, n = 26) | **57.7 %** (15/26), macro-F1 0.56 |
| 5×5 cross-validation on all 130 dishes | **75.7 % ± 6.4** |
| 28 new dishes written by hand (not in `dishes.csv`) | **78.6 %** |

The single test split is an unusually hard draw (worse than 98.5 % of random splits; see the notebook §A5 and the write-up), so I report
all three numbers rather than only the flattering one. Part B examples with their outputs are in `outputs/part_b_example_outputs.txt`.

## Repository layout

```
part_a_classifier.py     cleaning, split, two models, evaluation, predict_category()
part_b_sales_qa.py       loading/cleaning, question parser, pandas executor, charts, CLI
Kharchify_Interview_Task.ipynb   narrated run-through with outputs
tests/                   60 unit tests (Part B numbers checked against an independent csv-based oracle)
data/                    dishes.csv, sales.csv (as provided)
outputs/                 confusion matrix, charts, saved results
WRITEUP.md / .pdf        one-page write-up
```

## How Part B works

`question → parse → Query(intent, metric, period, dish/category, n) → pandas → answer`

The parser only decides **what** to compute; pandas does all the arithmetic, so numbers can't be hallucinated.
Anything it can't answer safely produces a polite message: off-topic questions, dishes not on the menu (with "Did you mean…?"),
periods outside the data, data the CSV doesn't have (profit, customers, order counts), future dates.

Supported: totals (revenue/units) overall, per dish, per group of dishes (`naan`) or category (`desserts`); top/bottom N dishes or categories
by units or revenue; best/worst day; averages per day; sales by category/day; dish price; periods like `today`, `yesterday`, `this/last week`,
`this/last month`, `last 7 days`, `5 nov`, `2026-11-05`, `last friday`, `in october`; misspelled dish names.

### Assumptions (also stated in answers where relevant)

* **"Today" = the last date in `sales.csv`** (Tue 17 Nov 2026) so the sample data gives reproducible answers whenever it is run.
  Use `--as-of today` (or `as_of=date.today()`) for a live system; if the real date is past the data, the assistant says "no data" instead of ₹0.
* Weeks run **Monday–Sunday**; "last N days" is a rolling window including today. Every answer prints the exact range it used.
* **Revenue = quantity × price** (price is a per-unit price: it is constant per dish in the file). Currency is shown as ₹ (assumed).
* Exact duplicate sales rows are kept (they could be separate orders) but counted in the cleaning report.

## Known limitations

* Part A has only 130 labelled dishes, so it struggles with words it has never seen and with labels that follow house conventions; it returns a confidence so low-confidence suggestions can be confirmed by the owner.
* Part B is rule-based, so unusual phrasings fall back to the polite message; a comparison of two periods in one question is not supported.
