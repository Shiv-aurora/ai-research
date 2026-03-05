import sys
from pathlib import Path
import yaml
import mlflow
import shutil

# Add src to path just in case
sys.path.append(str(Path.cwd()))

from src.utils.tracker import MLTracker


def test_project_structure():
    """Checks if key folders exist."""
    assert Path("conf/base/config.yaml").exists(), "Config missing!"
    assert Path("data/raw").exists(), "Data/Raw folder missing!"
    assert Path("src/__init__.py").exists(), "Source package missing!"
    print("✅ Project Structure: OK")


def test_config_loading():
    """Checks if config is readable."""
    with open("conf/base/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project_name"] == "rive", "Config content mismatch"
    print("✅ Config Loading: OK")


def test_mlflow_logging():
    """Checks if MLflow can actually write to disk."""
    import numpy as np
    
    # Clean up old test runs if any
    if Path("mlruns").exists():
        # Optional: clear it, or just append. Let's just append.
        pass
    
    tracker = MLTracker("titan_v8_smoke_test")
    
    # Start a run context for logging
    with tracker.start_run(run_name="smoke_test"):
        tracker.log_params({"test_param": 123})
        # log_metrics expects y_true and y_pred arrays
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        tracker.log_metrics(y_true, y_pred, step=1)
    
    # Check if experiment exists
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("titan_v8_smoke_test")
    assert exp is not None, "MLflow experiment not created"
    print("✅ MLflow Logging: OK")


if __name__ == "__main__":
    test_project_structure()
    test_config_loading()
    test_mlflow_logging()
    print("\n🚀 SMOKE TEST PASSED. Ready for Data Engine.")

