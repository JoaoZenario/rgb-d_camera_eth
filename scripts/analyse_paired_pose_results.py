import csv
import numpy as np

from collections import defaultdict

PATH = "outputs/paired_pose_experiment/per_trial_results.csv"

with open(PATH, newline="") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(dict)

for r in rows:
    groups[int(r["seed"])][r["condition"]] = r

conditions = ["ideal", "noise_only", "dropout_only", "full"]

print("\n========== SUCCESS THRESHOLD SENSITIVITY ==========")

for position_limit in [3, 5, 7, 10]:
    print(f"\nPosition threshold: {position_limit} mm")

    for condition in conditions:
        subset = [
            r[condition] for r in groups.values()
            if condition in r
        ]

        successes = [
            float(x["translation_error_mm"]) <= position_limit
            and float(x["rotation_error_deg"]) <= 5.0
            for x in subset
        ]

        print(
            f"{condition:12s}: "
            f"{np.mean(successes)*100:5.1f}%"
        )


print("\n========== PAIRED DIFFERENCES ==========")

for condition in ["noise_only", "dropout_only", "full"]:

    translation_deltas = []
    adds_deltas = []

    for seed, data in groups.items():
        if "ideal" not in data or condition not in data:
            continue

        ideal = data["ideal"]
        test = data[condition]

        translation_deltas.append(
            float(test["translation_error_mm"])
            - float(ideal["translation_error_mm"])
        )

        adds_deltas.append(
            float(test["adds_mm"])
            - float(ideal["adds_mm"])
        )

    print(f"\n{condition} minus ideal:")
    print(
        "Translation difference: "
        f"{np.mean(translation_deltas):+.3f} mm "
        f"(SD {np.std(translation_deltas, ddof=1):.3f})"
    )
    print(
        "ADD-S difference: "
        f"{np.mean(adds_deltas):+.3f} mm "
        f"(SD {np.std(adds_deltas, ddof=1):.3f})"
    )


print("\n========== TARGET DROPOUT RATE ==========")

for condition in ["dropout_only", "full"]:
    rates = []

    for data in groups.values():
        if condition not in data:
            continue

        valid = int(data[condition]["valid_target_pixels"])
        ideal = int(data["ideal"]["valid_target_pixels"])

        rates.append((ideal - valid) / ideal * 100)

    print(
        f"{condition}: mean {np.mean(rates):.2f}%, "
        f"range {np.min(rates):.2f}–{np.max(rates):.2f}%"
    )