import subprocess
import sys
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parent.parent / "run.py"

def _make_case(task_name: str, is_training: str) -> tuple[str]:
    return (
        "--task_name",      task_name,
        "--is_training",    is_training,
        "--root_path",      "./data/Public_Health",
        "--data_path",      "US_FLURATIO_Week.csv",
        "--model_id",       "Public_Health_2021_24_12_fullLLM_0",
        "--model",          "Informer",
        "--data",           "custom",
        "--features",       "M",
        "--seq_len",        "24",
        "--label_len",      "12",
        "--pred_len",       "12",
        "--des",            "Exp",
        "--seed",           "2021",
        "--type_tag",       "#F#",
        "--text_len",       "4",
        "--prompt_weight",  "0.1",
        "--pool_type",      "avg",
        "--save_name",      "result_health_bert",
        "--llm_model",      "BERT",
        "--huggingface_token", "NA",
        "--use_fullmodel",  "0",
    )

CASES = [
    _make_case("short_term_forecast", "1"),
    _make_case("long_term_forecast", "1"),
    _make_case("short_term_forecast", "0"),
    _make_case("long_term_forecast", "0"),
]

# TODO: maybe compare losses from output to ones in paper in code 
@pytest.mark.parametrize("extra_args", CASES)
def test_baseline(extra_args):
    """ Smoke-test run.py for basic repo functionality. 
        All example arguments taken from example script (week_health.sh). 
        Uses BERT and iTransformer.

    Args:
        extra_args (list):  Configs to test.
                            1. short term with training
                            2. long term with training
                            3. short term with no training
                            4. long term with no training
    """
    
    cmd = [sys.executable, str(SCRIPT), *extra_args]
    completed = subprocess.run(cmd, text=True)
    
    # Minimal assertion
    assert completed.returncode == 0, completed.stderr