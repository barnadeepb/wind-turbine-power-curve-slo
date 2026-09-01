"""Phase 3: Vertex AI (Gemini Enterprise Agent Platform) AutoML Tabular
regression — build-vs-buy baseline, run last and deliberately after every
local model has its own established result.

Cloud-side only; doesn't touch local CPU/GPU. Uses the same predefined
2016/2017 time-based split as every other model (via the ml_use column),
not AutoML's own random split, for a fair comparison.
"""

import json
import pathlib
import time

from google.cloud import aiplatform

PROJECT_ID = "project-0615c873-134d-4b53-b2e"
REGION = "us-central1"
BUCKET = "project-0615c873-134d-4b53-b2e-turbine-data"
GCS_CSV = f"gs://{BUCKET}/automl_combined.csv"
TARGET = "Prod_LatestAvg_TotActPwr"
BUDGET_MILLI_NODE_HOURS = 1000  # 1 node-hour, ~$21 at $21.25/node-hour

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

    print("creating tabular dataset from GCS CSV...", flush=True)
    dataset = aiplatform.TabularDataset.create(
        display_name="turbine-power-curve",
        gcs_source=GCS_CSV,
    )
    print(f"dataset created: {dataset.resource_name}", flush=True)

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="turbine-power-curve-automl",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse",
    )

    print(f"submitting training job, budget={BUDGET_MILLI_NODE_HOURS} milli-node-hours (~1 hour, ~$21)...", flush=True)
    t0 = time.time()
    model = job.run(
        dataset=dataset,
        target_column=TARGET,
        predefined_split_column_name="ml_use",
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
        model_display_name="turbine-power-curve-model",
        disable_early_stopping=False,
        sync=True,  # block until done — this call runs for close to the full budget
    )
    train_seconds = time.time() - t0
    print(f"training finished in {train_seconds:.0f}s, model: {model.resource_name}", flush=True)

    evals = list(model.list_model_evaluations())
    print(f"got {len(evals)} evaluation(s)", flush=True)
    eval_metrics = evals[0].metrics if evals else {}

    result = {
        "model_resource_name": model.resource_name,
        "train_seconds": train_seconds,
        "budget_milli_node_hours": BUDGET_MILLI_NODE_HOURS,
        "vertex_metrics": dict(eval_metrics),
    }

    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics" / "phase3_automl.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved results/metrics/phase3_automl.json", flush=True)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
