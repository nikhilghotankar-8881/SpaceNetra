import unittest
from pathlib import Path
from src.config import AppConfig, config


class TestConfig(unittest.TestCase):
    def test_default_config_loading(self):
        self.assertEqual(config.project.name, "SpaceNetra")
        self.assertEqual(config.project.version, "0.1.0")
        self.assertIn(config.system.device, ["auto", "cuda", "cpu"])
        self.assertIsInstance(config.paths.data_dir, Path)

    def test_custom_config_load(self):
        loaded = AppConfig.load()
        self.assertEqual(loaded.project.name, "SpaceNetra")
        self.assertEqual(loaded.change_detection.default_model, "siamese_unet")


if __name__ == "__main__":
    unittest.main()
