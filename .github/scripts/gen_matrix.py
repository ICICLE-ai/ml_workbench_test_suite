import json, sys
from pathlib import Path

# Add the repository root to the import path so we can import test_ct
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from test_ct import get_all_experiments, get_all_experiment_ids

matrix = []
for exp, full_id in zip(get_all_experiments(), get_all_experiment_ids()):
    model, device, site, dataset, mode = exp
    dataset_name = dataset["name"] if isinstance(dataset, dict) else str(dataset)

    k_id = full_id.split('@')[-1]

    matrix.append({
        "full_id": full_id,
        "k_id":    k_id,
        "model":   model,
        "device":  device,
        "site":    site,
        "dataset": dataset_name,
        "mode":    mode,
    })

print(json.dumps(matrix))
