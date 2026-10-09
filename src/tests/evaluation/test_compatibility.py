import json
import unittest
from types import SimpleNamespace

from agentgs.evaluation.metrics.compatibility import score_compatible


def spec(family, fields, units, **extra):
    return {"scoring_schema": dict(family=family, required_fields=fields, units=units, **extra)}


EARTH = spec(
    "earth_three_measurements",
    ["moho", "lid_bottom", "dlnvs"],
    {"moho": "km", "lid_bottom": "km", "dlnvs": "%"},
    expected_sample_depth_km=100.0,
)
PALEO = spec(
    "paleo_temperature_sea_level",
    ["age", "temperature_anomaly", "sea_level"],
    {"age": "kyr BP", "temperature_anomaly": "°C", "sea_level": "m"},
)
OCEAN = spec(
    "ocean_annual_summary",
    ["valid_count", "mean_fco2", "mean_sst"],
    {"valid_count": "count", "mean_fco2": "µatm", "mean_sst": "°C"},
)
MINERAL = spec(
    "mineral_location",
    ["site_name", "latitude", "longitude", "dep_id"],
    {"latitude": "degree", "longitude": "degree"},
)


def score(expected, predicted, schema, question="", kind="structured_record"):
    task = SimpleNamespace(expected_answer=expected, question=question, answer_type=kind)
    return score_compatible(task, predicted, schema)["normalized"]["score"]


class CompatibilityTests(unittest.TestCase):
    def test_equivalent_aliases_and_micro_symbols(self):
        self.assertEqual(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 350 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_json_fields(self):
        self.assertEqual(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                json.dumps({"valid_count": 12, "mean_fCO2": "350 µatm", "mean_SST": "16 °C"}),
                OCEAN,
            ),
            1,
        )

    def test_absolute_kelvin_conversion_for_sst(self):
        self.assertEqual(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 0.00035 atm; mean_SST: 289.15 K",
                OCEAN,
            ),
            1,
        )

    def test_temperature_anomaly_kelvin_has_no_offset(self):
        self.assertEqual(
            score(
                "age≈0.10 kyr BP; T≈-0.06 °C; SL≈-1.5 m",
                "age: 100 yr BP; temperature_anomaly: -0.06 K; sea_level: -150 cm",
                PALEO,
            ),
            1,
        )

    def test_dimension_mismatch_is_not_accepted(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 350 km; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_swapped_values_are_not_accepted(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 350; mean_fCO2: 12 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_wrong_labels_do_not_borrow_units_from_other_fields(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 16 °C; mean_SST: 350 µatm",
                OCEAN,
            ),
            1,
        )

    def test_missing_field_is_not_completed(self):
        self.assertAlmostEqual(
            score(
                "age≈0.10 kyr BP; T≈-0.06 °C; SL≈-1.5 m",
                "temperature_anomaly: -0.06 °C; sea_level: -1.5 m",
                PALEO,
            ),
            2 / 3,
            places=6,
        )

    def test_unknown_field_does_not_become_zero(self):
        self.assertAlmostEqual(
            score(
                "age≈0.10 kyr BP; T≈0.00 °C; SL≈-1.5 m",
                "age: 0.1 kyr BP; temperature_anomaly: unknown; sea_level: -1.5 m",
                PALEO,
            ),
            2 / 3,
            places=6,
        )

    def test_conflicting_duplicate_rejected(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12; valid_count: 99; mean_fCO2: 350 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_negative_count_rejected(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "valid_count: -12; mean_fCO2: 350 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_negated_result_rejected(self):
        self.assertLess(
            score(
                "12 records; 350.00 μatm; 16.00 °C",
                "not valid_count: 12; mean_fCO2: 350 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_thousands_separator_preserved_as_number(self):
        self.assertEqual(
            score(
                "12000 records; 350.00 μatm; 16.00 °C",
                "valid_count: 12,000; mean_fCO2: 350 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_depth_context_not_an_extra_answer(self):
        self.assertEqual(
            score(
                "Moho 20.00 km; LID-BOTTOM 70.00 km; dlnVs(100 km) 2.00%",
                "Moho: 20 km; LID-BOTTOM: 70 km; dlnVs: 2 %",
                EARTH,
            ),
            1,
        )

    def test_metre_and_ratio_conversion(self):
        self.assertEqual(
            score(
                "Moho 20.00 km; LID-BOTTOM 70.00 km; dlnVs(100 km) 2.00%",
                "Moho: 20000 m; LID-BOTTOM: 70000 m; dlnVs: 0.02 fraction",
                EARTH,
            ),
            1,
        )

    def test_wrong_explicit_depth_is_not_accepted(self):
        self.assertLess(
            score(
                "Moho 20.00 km; LID-BOTTOM 70.00 km; dlnVs(100 km) 2.00%",
                "Moho: 20 km; LID-BOTTOM: 70 km; dlnVs(660 km): 2 %",
                EARTH,
            ),
            1,
        )

    def test_unrequested_epicenter_does_not_dilute_named_results(self):
        self.assertEqual(
            score(
                "Moho 20.00 km; LID-BOTTOM 70.00 km; dlnVs(100 km) 2.00%",
                "Epicenter: 2.3, 93.1; Moho: 20 km; LID-BOTTOM: 70 km; dlnVs at 100 km: 2 %",
                EARTH,
            ),
            1,
        )

    def test_coordinates_keep_decimal_comma_boundary(self):
        self.assertEqual(
            score(
                "Example Prospect; (12.340, -56.780); dep_id=999",
                "site_name: Example Prospect; latitude: 12.34; longitude: -56.78; dep_id: 999",
                MINERAL,
                kind="multi_text",
            ),
            1,
        )

    def test_coordinate_order_not_interchangeable(self):
        self.assertLess(
            score(
                "Example Prospect; (12.340, -56.780); dep_id=999",
                "site_name: Example Prospect; latitude: -56.78; longitude: 12.34; dep_id: 999",
                MINERAL,
                kind="multi_text",
            ),
            1,
        )

    def test_missing_identifier_remains_missing(self):
        self.assertEqual(
            score(
                "Example Prospect; (12.340, -56.780); dep_id=999",
                "site_name: Example Prospect; latitude: 12.34; longitude: -56.78",
                MINERAL,
                kind="multi_text",
            ),
            0.75,
        )

    def test_text_prefix_is_not_identity(self):
        self.assertLess(
            score(
                "Example Prospect; (12.340, -56.780); dep_id=999",
                "site_name: Example; latitude: 12.34; longitude: -56.78; dep_id: 999",
                MINERAL,
                kind="multi_text",
            ),
            1,
        )

    def test_wrong_identifier_not_inferred_from_question(self):
        self.assertLess(
            score(
                "Example Prospect; (12.340, -56.780); dep_id=999",
                "site_name: Example Prospect; latitude: 12.34; longitude: -56.78; dep_id: 888",
                MINERAL,
                "Look up dep_id=999",
                kind="multi_text",
            ),
            1,
        )

    def test_E_choice_supported(self):
        self.assertEqual(
            score("E", "Answer: E", {"allowed_options": list("ABCDE")}, kind="choice"), 1
        )

    def test_ambiguous_choice_rejected(self):
        self.assertEqual(
            score("E", "A or E", {"allowed_options": list("ABCDE")}, kind="choice"), 0
        )

    def test_correct_rounded_value_accepted(self):
        self.assertEqual(
            score(
                "12 records; 350.50 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 350.504 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_outside_rounding_tolerance_rejected(self):
        self.assertLess(
            score(
                "12 records; 350.50 μatm; 16.00 °C",
                "valid_count: 12; mean_fCO2: 350.52 µatm; mean_SST: 16 °C",
                OCEAN,
            ),
            1,
        )

    def test_unparseable_reference_requires_review(self):
        task = SimpleNamespace(
            expected_answer="ambiguous reference", question="", answer_type="structured_record"
        )
        result = score_compatible(task, "valid_count: 12", OCEAN)
        self.assertIsNone(result["normalized"])
        self.assertTrue(result["needs_review"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
