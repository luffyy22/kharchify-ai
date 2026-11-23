"""Tests for Part A. Run from the repo root:  python -m unittest discover -s tests -v"""
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import part_a_classifier as A  # noqa: E402

warnings.filterwarnings("ignore")


class TestCleaning(unittest.TestCase):
    def test_messy_input_is_cleaned(self):
        raw = pd.DataFrame({
            "dish_name": ["  Paneer   Tikka ", "paneer tikka", "Masala Chai", "Masala Chai", "Kheer", None, "  ", "Dal Tadka"],
            "category": ["starter", "Starter ", "Beverage", "Dessert", "desserts", "Bread", "Bread", None],
        })
        df, rep = A.clean_dishes(raw)
        self.assertEqual(rep["missing_dropped"], 3)                 # None name, blank name, None category
        self.assertEqual(rep["duplicates_dropped"], 1)              # the two Paneer Tikka rows
        self.assertEqual(rep["conflicting_labels_dropped"], 2)      # Masala Chai is Beverage AND Dessert
        self.assertEqual(rep["conflicting_examples"], ["Masala Chai"])
        self.assertEqual(sorted(df["dish_name"]), ["Kheer", "Paneer Tikka"])
        self.assertEqual(dict(zip(df["dish_name"], df["category"])), {"Paneer Tikka": "Starter", "Kheer": "Dessert"})

    def test_real_file_loads_clean(self):
        df, rep = A.load_dishes()
        self.assertEqual(len(df), 130)
        self.assertEqual(set(df["category"]), set(A.CATEGORIES))
        self.assertFalse(df.isna().any().any())


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.df, _ = A.load_dishes()

    def test_split_is_disjoint_stratified_and_reproducible(self):
        Xtr, Xte, ytr, yte = A.split_data(self.df)
        self.assertEqual(len(Xtr) + len(Xte), len(self.df))
        self.assertFalse(set(Xtr) & set(Xte))                       # no leakage
        self.assertEqual(set(yte), set(A.CATEGORIES))               # every class is tested
        Xtr2, Xte2, _, _ = A.split_data(self.df)
        self.assertEqual(list(Xte), list(Xte2))                     # fixed seed

    def test_evaluation_never_uses_training_rows(self):
        Xtr, Xte, ytr, yte = A.split_data(self.df)
        model = A.make_model("word+char").fit(Xtr, ytr)
        res = A.evaluate(model, Xte, yte)
        self.assertEqual(len(res["pred"]), len(Xte))
        self.assertEqual(res["confusion"].sum(), len(Xte))


class TestPrediction(unittest.TestCase):
    def test_predict_category_returns_a_known_category(self):
        for name in ["Butter Garlic Naan", "Mango Milkshake", "zzzz qqqq", "  Chicken   Lollipop  ", "Chicken 65 Special"]:
            self.assertIn(A.predict_category(name), A.CATEGORIES)

    def test_obvious_cases(self):
        self.assertEqual(A.predict_category("Butter Garlic Naan"), "Bread")
        self.assertEqual(A.predict_category("Veg Dum Biryani"), "Rice")
        self.assertEqual(A.predict_category("Chicken Soup"), "Starter")

    def test_deterministic(self):
        self.assertEqual(A.predict_with_confidence("Tiramisu"), A.predict_with_confidence("Tiramisu"))

    def test_bad_input_raises_clear_error(self):
        for bad in ["", "   ", None, 5, "!!!"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                A.predict_category(bad)

    def test_confidence_is_a_probability_and_flags_unsure(self):
        r = A.predict_with_confidence("Cappuccino")
        self.assertTrue(0 < r["confidence"] <= 1)
        self.assertAlmostEqual(sum(p for _, p in A.predict_with_confidence("Cappuccino", top_k=6)["top_k"]), 1.0, places=6)
        self.assertEqual(r["needs_confirmation"], r["confidence"] < A.LOW_CONFIDENCE)

    def test_unseen_dishes_really_are_unseen(self):
        df, _ = A.load_dishes()
        chk = A.unseen_check(A.get_production_model(), df["dish_name"])   # asserts no overlap inside
        self.assertGreaterEqual(len(chk), 20)


class TestReproducibility(unittest.TestCase):
    def test_same_numbers_every_run(self):
        df, _ = A.load_dishes()
        Xtr, Xte, ytr, yte = A.split_data(df)
        a = A.evaluate(A.make_model("word+char").fit(Xtr, ytr), Xte, yte)["accuracy"]
        b = A.evaluate(A.make_model("word+char").fit(Xtr, ytr), Xte, yte)["accuracy"]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 15 / 26)                           # number quoted in the write-up


if __name__ == "__main__":
    unittest.main()
