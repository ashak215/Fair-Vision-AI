from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.analytics import (  # noqa: E402
    build_detailed_gender_analytics,
    build_gender_summary,
    get_promotion_summary,
    get_salary_summary,
)
from utils.data_loader import load_employee_data  # noqa: E402
from utils.ml_model import train_promotion_model  # noqa: E402
from utils.reports import (  # noqa: E402
    create_ml_report,
    create_summary_report,
    serialize_report_csv,
)


class FairVisionCoreTests(unittest.TestCase):
    def test_bundled_dataset_and_analytics(self) -> None:
        frame, quality, source = load_employee_data()
        self.assertEqual(len(frame), 60)
        self.assertEqual(quality["duplicates_removed"], 0)
        self.assertEqual(source, "Bundled synthetic demonstration dataset")
        self.assertEqual(frame["gender"].value_counts().to_dict(), {"Female": 30, "Male": 30})
        self.assertEqual(len(get_salary_summary(frame)), 2)
        self.assertEqual(len(get_promotion_summary(frame)), 2)
        self.assertEqual(len(build_gender_summary(frame)), 30)
        self.assertEqual(len(build_detailed_gender_analytics(frame)), 30)

    def test_upload_normalization_and_quality_checks(self) -> None:
        csv = StringIO(
            "gender,age,department,experience_years,salary,promotion_status\n"
            "F,29,Data,5,60000,Promoted\n"
            "X,not-a-number,Ops,-1,-5,Maybe\n"
            "F,29,Data,5,60000,Promoted\n"
        )
        frame, quality, source = load_employee_data(uploaded_file=csv)
        self.assertEqual(source, "Uploaded CSV — provenance not independently verified")
        self.assertEqual(len(frame), 2)
        self.assertEqual(quality["duplicates_removed"], 1)
        self.assertEqual(quality["invalid_numeric_values"], 3)
        self.assertEqual(quality["unexpected_gender_labels"], 1)
        self.assertEqual(quality["unexpected_promotion_labels"], 1)
        self.assertEqual(frame.loc[1, "gender"], "Other/Unknown")
        self.assertEqual(frame.loc[1, "promotion_status"], "Unknown")
        self.assertTrue(pd.isna(frame.loc[1, "age"]))
        self.assertTrue(pd.isna(frame.loc[1, "experience_years"]))
        self.assertTrue(pd.isna(frame.loc[1, "salary"]))

    def test_model_requires_adequate_target_and_excludes_gender(self) -> None:
        frame, _, _ = load_employee_data()
        result = train_promotion_model(frame)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertNotIn("gender", result["feature_columns"])
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), result["test_size"])
        for metric in ("accuracy", "precision", "recall", "f1"):
            self.assertGreaterEqual(result[metric], 0)
            self.assertLessEqual(result[metric], 1)

        too_small = train_promotion_model(frame.head(12))
        self.assertFalse(too_small["available"])
        self.assertIn("At least", too_small["reason"])

    def test_reports_are_downloadable_and_csv_carries_metadata(self) -> None:
        frame, _, source = load_employee_data()
        model = train_promotion_model(frame)
        filters = {
            "departments": ["Data"],
            "experience": (1, 20),
        }
        summary_pdf = create_summary_report(frame, source, filters)
        ml_pdf = create_ml_report(
            model,
            source,
            filters={"departments": [], "experience": ("all", "all")},
        )
        self.assertTrue(summary_pdf.startswith(b"%PDF"))
        self.assertTrue(ml_pdf.startswith(b"%PDF"))

        exported = serialize_report_csv(frame.head(1), source, filters)
        csv_frame = pd.read_csv(StringIO(exported.decode("utf-8")))
        self.assertEqual(csv_frame.loc[0, "data_source"], source)
        self.assertIn("Departments: Data", csv_frame.loc[0, "applied_filters"])
        self.assertIn("human review", csv_frame.loc[0, "limitations"].casefold())


if __name__ == "__main__":
    unittest.main()
