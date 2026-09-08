"""Tests for the local-only ComfyUI restart support."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_sai_nodes"
if PACKAGE_NAME not in sys.modules:
    PACKAGE_SPEC = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
    PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
    sys.modules[PACKAGE_NAME] = PACKAGE
    PACKAGE_SPEC.loader.exec_module(PACKAGE)

from comfyui_sai_nodes import WEB_DIRECTORY
from comfyui_sai_nodes.server.restart import (
    RESTART_EXIT_CODE,
    is_loopback_address,
)


class RestartSupportTests(unittest.TestCase):
    def test_accepts_only_loopback_addresses(self) -> None:
        self.assertTrue(is_loopback_address("127.0.0.1"))
        self.assertTrue(is_loopback_address("::1"))
        self.assertTrue(is_loopback_address("::ffff:127.0.0.1"))
        self.assertTrue(is_loopback_address("localhost"))
        self.assertFalse(is_loopback_address("192.168.1.10"))
        self.assertFalse(is_loopback_address(None))

    def test_supervisor_contract_uses_dedicated_exit_code(self) -> None:
        launcher = (PACKAGE_ROOT / "tools" / "dev" / "start_dev_comfyui.bat").read_text(
            encoding="utf-8"
        )

        self.assertEqual(RESTART_EXIT_CODE, 75)
        self.assertIn('if "%EXIT_CODE%"=="75"', launcher)
        self.assertIn("goto run_comfyui", launcher)

    def test_frontend_directory_is_exported(self) -> None:
        self.assertEqual(WEB_DIRECTORY, "./web")
        self.assertTrue((PACKAGE_ROOT / "web" / "restart.js").is_file())


if __name__ == "__main__":
    unittest.main()
