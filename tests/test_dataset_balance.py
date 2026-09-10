import unittest

from spatialforge.embodied import dataset_balance as db


def _row(i, episode, answer, house="h0", category="mug"):
    return {
        "id": f"{episode}-s{i}",
        "episode_id": episode,
        "answer": answer,
        "scene_id": house,
        "category": category,
    }


class DatasetBalanceTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.outcomes = {}
        for ep, answer, success, code in (
            ("ep-s1", "MoveAhead", True, "success"),
            ("ep-s2", "RotateLeft", True, "success"),
            ("ep-f1", "MoveAhead", False, "Done emitted but no target-category object is visible."),
            ("ep-f2", "MoveAhead", False, "unreachable_task"),
        ):
            self.outcomes[ep] = {
                "episode_id": ep, "success": success, "outcome_code": code,
                "steps": 5, "category": "mug", "house": "h0",
            }
            for i in range(4):
                self.rows.append(_row(i, ep, answer))

    def test_classify_core_aux_dropped(self):
        classified = db.classify_rows(self.rows, self.outcomes)
        self.assertEqual(len(classified["core"]), 8)
        self.assertEqual(len(classified["aux"]), 4)
        self.assertEqual(len(classified["dropped"]), 4)
        self.assertTrue(all(r["row_class"] == "core" for r in classified["core"]))

    def test_select_success_core_bounds_aux(self):
        classified = db.classify_rows(self.rows, self.outcomes)
        selected, info = db.select_success_core(classified, aux_failure_ratio=0.25)
        self.assertEqual(info["core_rows"], 8)
        self.assertEqual(info["aux_failure_rows_selected"], 2)
        self.assertEqual(len(selected), 10)

    def test_episode_balance_caps_per_episode(self):
        balanced, info = db.apply_balance(self.rows, mode="episode", episode_cap=2)
        self.assertEqual(len(balanced), 8)
        self.assertEqual(info["episodes"], 4)

    def test_action_balanced_downsamples_majority(self):
        rows = []
        for i in range(100):
            rows.append(_row(i, "ep-a", "MoveAhead"))
        for i in range(20):
            rows.append(_row(i, "ep-b", "RotateLeft"))
        for i in range(20):
            rows.append(_row(i, "ep-c", "Done"))
        balanced, info = db.apply_balance(
            rows, mode="action_balanced", min_class_count=5,
        )
        counts = db._counts(balanced)
        self.assertEqual(counts["RotateLeft"], 20)
        self.assertEqual(counts["Done"], 20)
        self.assertEqual(counts["MoveAhead"], info["target_per_class"])
        self.assertLess(counts["MoveAhead"], 100)
        # no duplication: all kept ids unique
        ids = [r["id"] for r in balanced]
        self.assertEqual(len(ids), len(set(ids)))

    def test_action_weighted_emits_weights(self):
        _, info = db.apply_balance(self.rows, mode="action_weighted")
        self.assertIn("MoveAhead", info["class_weights"])
        self.assertGreater(info["class_weights"]["MoveAhead"], 0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            db.apply_balance(self.rows, mode="magic")


if __name__ == "__main__":
    unittest.main()
