import json
import numpy as np
from scipy.stats import spearmanr

TAILS = [32, 64, 128, 256]

all_data = {}

for T in TAILS:
    path = f"analysis/results/refined_tail_{T}.json"

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    all_data[T] = {x["id"]: x for x in data}


print("=" * 100)
print("ID | LABEL | MEAN | MAX | RANGE | SLOPE | D128-D32")
print("=" * 100)

rows = []

for i in sorted(all_data[128]):

    values = []

    for T in TAILS:
        s = all_data[T][i].get("sp_hid_tail")

        if s is None:
            values.append(np.nan)
        else:
            values.append(float(s))

    values = np.array(values)

    if np.isnan(values).any():
        continue

    mean = values.mean()
    maximum = values.max()
    minimum = values.min()
    value_range = maximum - minimum

    slope = np.polyfit(TAILS, values, 1)[0]

    delta = values[2] - values[0]

    label = all_data[128][i]["label"]

    print(
        f"{i:2d} | "
        f"{label:22s} | "
        f"{mean:.6f} | "
        f"{maximum:.6f} | "
        f"{value_range:.6f} | "
        f"{slope:.8f} | "
        f"{delta:.6f}"
    )

    rows.append({
        "id": i,
        "label": label,
        "mean": mean,
        "max": maximum,
        "range": value_range,
        "slope": slope,
        "delta128_32": delta,
    })


print()
print("=" * 100)
print("GROUP SUMMARY")
print("=" * 100)

for feature in ["mean", "max", "range", "slope", "delta128_32"]:

    h = [
        r[feature]
        for r in rows
        if "harmful" in r["label"].lower()
    ]

    b = [
        r[feature]
        for r in rows
        if "benign" in r["label"].lower()
    ]

    print(
        f"{feature:15s} "
        f"H={np.mean(h): .6f} "
        f"B={np.mean(b): .6f} "
        f"gap={np.mean(h)-np.mean(b): .6f}"
    )
