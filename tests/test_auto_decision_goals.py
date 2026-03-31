import unittest

from core.auto_decision_goals import apply_goal_preset_to_settings, find_goal_preset, get_auto_decision_goal_presets


class TestAutoDecisionGoals(unittest.TestCase):
    def test_presets_exist_and_have_unique_keys(self) -> None:
        presets = get_auto_decision_goal_presets()
        self.assertTrue(presets)
        keys = [p.key for p in presets]
        self.assertEqual(len(keys), len(set(keys)))

    def test_find_preset(self) -> None:
        p = find_goal_preset("balanced")
        self.assertIsNotNone(p)
        self.assertEqual(p.key, "balanced")

    def test_apply_goal_preset_updates_settings(self) -> None:
        out = apply_goal_preset_to_settings("weighted_balanced", current_settings={"ranking_objective": "rs_first"})
        self.assertEqual(out["ranking_objective"], "weighted_combined")
        self.assertIn("ranking_weights", out)
        self.assertTrue(out["ranking_weights"])

    def test_unknown_preset_noop(self) -> None:
        base = {"ranking_objective": "spec_pass_first"}
        out = apply_goal_preset_to_settings("does_not_exist", current_settings=base)
        self.assertEqual(out, base)


if __name__ == "__main__":
    unittest.main()

