import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


class NotebookShellEnvironmentTests(unittest.TestCase):
    def test_notebook_runners_activate_conda_environment(self):
        shell_runners = [
            NOTEBOOKS_DIR / "run_financial_q_and_a.sh",
            NOTEBOOKS_DIR / "run_fina_q_and_a.sh",
            NOTEBOOKS_DIR / "run_reddit_250k_analysis.sh",
        ]

        for script_path in shell_runners:
            content = script_path.read_text(encoding="utf-8")
            self.assertIn('ENV_NAME="${FINSIGHTAI_CONDA_ENV_NAME:-finsightai-notebooks}"', content)
            self.assertIn('eval "$(conda shell.bash hook)"', content)
            self.assertIn('conda activate "${ENV_NAME}"', content)

    def test_setup_script_uses_environment_yaml_and_requirements(self):
        content = (NOTEBOOKS_DIR / "setup_conda_env.sh").read_text(encoding="utf-8")
        self.assertIn('ENV_FILE="${SCRIPT_DIR}/environment.yml"', content)
        self.assertIn('REQ_FILE="${SCRIPT_DIR}/requirements.txt"', content)
        self.assertIn('conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune', content)
        self.assertIn('conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"', content)
        self.assertIn('python -m pip install -r "${REQ_FILE}"', content)


if __name__ == "__main__":
    unittest.main()
