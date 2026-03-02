# USAGE:
#   set GITHUB_TOKEN=... (cmd)   or   $env:GITHUB_TOKEN="..." (PowerShell)
#   set GITHUB_REPOSITORY=user/repo
#   python .github/scripts/check_and_issue.py [launcher_name]

import os, json, sys
from github import Github 

with open("metrics/rag_eval.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# List of actually used metrics in rag_eval.json
metric_keys = [
    "weighted_mae_score",
    "mean_faithfulness_score",
    "ragas_correctness",
    "ragas_answer_relevancy"
]

# Default thresholds (add more if needed)
default_thresholds = {
    "weighted_mae_score": ("max", float(os.getenv("THRESHOLD_MAE", "1.0"))),
    "mean_faithfulness_score": ("min", float(os.getenv("THRESHOLD_FAITHFULNESS", "0.5"))),
    "ragas_correctness": ("min", 0.5),
    "ragas_answer_relevancy": ("min", 0.5)
    # You can add more custom thresholds here
}

# Get launcher name from command line or environment
if len(sys.argv) > 1:
    launcher = sys.argv[1]
else:
    launcher = os.getenv("LAUNCHER_NAME", "daily check")

errors = []
for key in metric_keys:
    value = data.get(key)
    if value is None:
        errors.append(f"{key} missing")
        continue
    # If the metric has a defined threshold, check it
    if key in default_thresholds:
        mode, threshold = default_thresholds[key]
        if mode == "min" and value < threshold:
            errors.append(f"{key} {value:.4f} < min {threshold}")
        elif mode == "max" and value > threshold:
            errors.append(f"{key} {value:.4f} > max {threshold}")

if errors:
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(os.environ["GITHUB_REPOSITORY"])
    title = f"[ALERT] Evaluation: Metrics below threshold ({launcher})"
    body = f"Launcher: {launcher}\n\nThe following metrics are below threshold:\n\n" + "\n".join(f"- {e}" for e in errors)
    repo.create_issue(title=title, body=body)
    print("Issue created.")
else:
    print("All metrics above threshold.")
