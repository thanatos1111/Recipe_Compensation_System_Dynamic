import importlib
import unittest


CORE_MODULES = [
    "core.material_manager",
    "core.importer",
    "core.schemas",
    "core.labeling",
    "core.validation",
    "core.trend_fitting",
    "core.feature_engineering",
    "core.response_models",
    "core.instance_correction",
    "core.candidate_generator",
    "core.constraints",
    "core.optimizer",
    "core.safe_band",
    "core.update_pipeline",
    "core.exports",
]


class TestImports(unittest.TestCase):
    def test_core_imports(self) -> None:
        for mod_name in CORE_MODULES:
            with self.subTest(mod_name=mod_name):
                importlib.import_module(mod_name)


if __name__ == "__main__":
    unittest.main()

