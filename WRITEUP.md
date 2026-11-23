# Kharchify AI/ML Intern Task — Write-up

## What I built and why

**Part A – Dish category classifier.** I cleaned the data (trim/collapse spaces, normalise case and category spellings, drop blanks, duplicates and dishes with conflicting labels; the sample file was already clean), made a stratified 80/20 split (seed 42) and trained TF-IDF + Logistic Regression. I compared word n-grams (baseline) with word + character n-grams using cross-validation **on the training set only**; character n-grams won (CV 76.5 % vs 71.2 %) because they give partial credit to unseen or misspelled words, so that is the final model. `predict_category()` also returns a confidence so the UI can ask the owner to confirm uncertain suggestions.

**Part B – Sales Q&A.** A rule-based parser turns the question into a structured query (intent, period, dish/category, metric, n) and **pandas computes the answer**. I chose rules over an LLM because the question family is small, it needs no API key, it is deterministic, and it cannot invent numbers. The same structured query is what an LLM would be asked to output, so the parser can be swapped later. Unanswerable questions (off-topic, unknown dish, dates outside the data, profit/customers/order counts, the future) get a polite, specific message instead of a guess. Answers always state the date range used, any spelling correction, ties and partial-coverage warnings. 60 tests check the numbers against an independent `csv`-based oracle, and a fuzz test confirms it never crashes.

## Accuracy and what limits it

* **Held-out test accuracy: 57.7 % (15/26), macro-F1 0.56.** I do not think this is the model's true level: with 26 rows the 95 % interval is 39–74 %, and seed 42 turned out to be worse than 98.5 % of 200 random splits (both *chai* dishes landed in the test set, so the word never appeared in training). I kept the seed I had fixed in advance rather than choosing a luckier one.
* **More reliable estimates:** 5×5 cross-validation on all 130 dishes **75.7 % ± 6.4**; 28 new hand-labelled dishes **78.6 %**, where every mistake had confidence < 0.5.
* **Limits:** (1) tiny data — 65 % of dishes contain a word not seen in training, and accuracy is 71 % for those vs 89 % otherwise; (2) shared ambiguous words (*masala, chicken, paneer*) — Starter↔Main Course is 11 of the 30 cross-validated errors; (3) unique dessert names (*Kulfi, Falooda, Kheer*) give desserts only 50 % recall; (4) labels follow house conventions (Tandoori Chicken = Starter, Butter Chicken = Main).

## Assumptions

"Today" is the last date in `sales.csv` (17 Nov 2026) so the sample data answers reproducibly (`--as-of today` for live use); weeks run Monday–Sunday, so "this week" is only two days old and the answer says so; revenue = quantity × price; currency shown as ₹ (the file has no currency).

## With more time or data

* Part A: several hundred labelled dishes across many restaurants; pretrained text embeddings or an LLM few-shot prompt for words never seen; per-restaurant fine-tuning from owners' confirmations; a larger, repeated evaluation instead of one 26-row split.
* Part B: more phrasings and comparisons ("this week vs last week"); an LLM front-end that only outputs a validated query JSON while pandas still does all arithmetic; order IDs and timestamps in the data to support order counts and peak hours.
