import os

class ExperimentTracker:
    """
    Simple experiment tracker for logging results and parameters.
    Can be extended to use MLflow, wandb, or other tracking tools.
    """

    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_run = None
        self.run_id = None

    def start_run(self, run_name=None, params=None):
        """
        Start a new experiment run.
        Args:
            run_name (str): Optional name for the run.
            params (dict): Optional dictionary of parameters to log.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_name or f"run_{timestamp}"
        self.current_run = {
            "run_id": self.run_id,
            "params": params or {},
            "metrics": {},
            "artifacts": []
        }
        print(f"[ExperimentTracker] Started run: {self.run_id}")

    def log_param(self, key, value):
        if self.current_run is not None:
            self.current_run["params"][key] = value

    def log_params(self, params):
        if self.current_run is not None and isinstance(params, dict):
            self.current_run["params"].update(params)

    def log_metric(self, key, value):
        if self.current_run is not None:
            self.current_run["metrics"][key] = value

    def log_metrics(self, metrics):
        if self.current_run is not None and isinstance(metrics, dict):
            self.current_run["metrics"].update(metrics)

    def log_artifact(self, file_path):
        if self.current_run is not None and os.path.exists(file_path):
            self.current_run["artifacts"].append(file_path)

    def end_run(self):
        """
        Save the run information to a log file.
        """
        if self.current_run is not None:
            import json
            log_path = os.path.join(self.log_dir, f"{self.run_id}.json")
            with open(log_path, "w") as f:
                json.dump(self.current_run, f, indent=2)
            print(f"[ExperimentTracker] Run saved to {log_path}")
            self.current_run = None
            self.run_id = None

# Example usage
if __name__ == "__main__":
    # Set up experiment tracker
    tracker = ExperimentTracker()

    # Start a new run
    tracker.start_run(run_name="auair_baseline", params={
        "dataset": "AU-AIR",
        "image_folder": "data/images",
        "annotation_file": "data/annotations.json"
    })

    # Log some metrics (dummy values for illustration)
    tracker.log_metric("mAP", 0.72)
    tracker.log_metric("loss", 0.15)

    # Log an artifact (e.g., model checkpoint or result file)
    # tracker.log_artifact("checkpoints/model.pt")

    # End the run and save results
    tracker.end_run()
