"""Headless checks that the UI stays separate from the game and the bots.

Training must never depend on the UI: nothing outside ``ui/`` (and the UI's
own tests) may import ``ui``, importing the engine and the bots must not
load pygame, and the UI's logic modules must not load pygame either.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".venv", "venv", ".git", "__pycache__", "ui"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )


class TestIsolation(unittest.TestCase):
    def test_nothing_outside_ui_imports_ui(self):
        offenders = []
        for path in ROOT.rglob("*.py"):
            parts = set(path.relative_to(ROOT).parts[:-1])
            if parts & EXCLUDED_DIRS or path.name.startswith("test_ui_"):
                continue
            if any(m == "ui" or m.startswith("ui.") for m in imported_modules(path)):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_engine_and_bots_do_not_load_pygame_or_ui(self):
        result = run_python(
            "import sys\n"
            "import game.environment, agents, agents.match, agents.evaluate\n"
            "assert 'pygame' not in sys.modules, 'pygame loaded'\n"
            "assert not any(m == 'ui' or m.startswith('ui.') for m in sys.modules)\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ui_logic_modules_do_not_load_pygame(self):
        result = run_python(
            "import sys\n"
            "import ui.config, ui.palette, ui.art, ui.font, ui.clock, ui.menu\n"
            "import ui.session, ui.scene\n"
            "assert 'pygame' not in sys.modules, 'pygame loaded'\n"
            "print('ok')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
