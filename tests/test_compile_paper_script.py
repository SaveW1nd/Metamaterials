from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "compile_paper.sh"


class TestCompilePaperScript(unittest.TestCase):
    def test_compile_succeeds_without_texbin_in_path(self) -> None:
        env = os.environ.copy()
        env["PATH"] = ":".join(
            entry for entry in env.get("PATH", "").split(":") if entry != "/Library/TeX/texbin"
        )

        result = subprocess.run(
            [str(SCRIPT)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"compile_paper.sh should succeed without relying on interactive shell PATH.\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
