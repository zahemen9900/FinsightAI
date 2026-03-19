import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class NotebookScriptSmokeTests(unittest.TestCase):
    def test_fina_script_help(self):
        script = REPO_ROOT / "notebooks" / "fina_q_and_a.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Inspect the Financial-QA CSV", result.stdout)

    def test_reddit_script_help(self):
        script = REPO_ROOT / "notebooks" / "reddit_250k_analysis.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Run Reddit finance JSONL analysis", result.stdout)

    def test_fina_script_missing_file_fails_fast(self):
        from notebooks.fina_q_and_a import main

        with self.assertRaises(FileNotFoundError):
            main(["--dataset-path", "/tmp/does-not-exist.csv"])

    def test_reddit_script_missing_file_fails_fast(self):
        from notebooks.reddit_250k_analysis import main

        with self.assertRaises(FileNotFoundError):
            main(["--input-jsonl", "/tmp/does-not-exist.jsonl"])


if __name__ == "__main__":
    unittest.main()
