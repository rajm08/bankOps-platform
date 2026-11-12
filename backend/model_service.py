from pathlib import Path
import joblib

class ModelService:
    ARTIFACTS = [
        "anomaly_detection.pkl",
        "banking_log_clustering_model.pkl",
        "incident_criticality.pkl",
        "timeseries_forecasting.pkl",
    ]

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
        self.loaded = {}
        self.errors = {}
        for filename in self.ARTIFACTS:
            path = tasks_dir / filename
            if not path.exists():
                self.errors[filename] = "file not found"
                continue
            try:
                self.loaded[filename] = joblib.load(path)
            except Exception as exc:
                self.errors[filename] = str(exc)

    def status(self):
        return {
            "loaded": list(self.loaded.keys()),
            "errors": self.errors,
        }

    def describe(self, filename: str):
        obj = self.loaded.get(filename)
        if obj is None:
            return {"loaded": False, "error": self.errors.get(filename, "not loaded")}
        if isinstance(obj, dict):
            return {
                "loaded": True,
                "type": type(obj).__name__,
                "keys": list(obj.keys()),
                "values": {
                    str(k): type(v).__name__
                    for k, v in obj.items()
                },
            }
        return {
            "loaded": True,
            "type": type(obj).__name__,
            "has_predict": callable(getattr(obj, "predict", None)),
        }
