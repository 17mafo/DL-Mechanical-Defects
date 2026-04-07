import pandas as pd
from pathlib import Path

# Path to folder containing CSV files
csv_dir = Path("C:/Users/Marti/Documents/DL-Mechanical-Defects/model_training/histories")

# Build one row per CSV with its name and mean metrics
rows = []
for csv_file in csv_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)
    name = csv_file.stem

    focus = None
    if "_1_" in name:
        focus = 1
    elif "_2_" in name:
        focus = 2

    preprocess = None
    for preprocess_name in ["initial", "outer_rim", "background"]:
        if preprocess_name in name:
            preprocess = preprocess_name
            break

    model_name = name
    if preprocess is not None:
        split_token = f"_{preprocess}_"
        if split_token in name:
            model_name = name.split(split_token, 1)[0]

    rows.append(
        {
            "name": model_name,
            "focus": focus,
            "preprocess": preprocess,
            "accuracy": df["accuracy"].mean(),
            "f1": df["f1"].mean(),
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
        }
    )

metrics_table = pd.DataFrame(rows)

preprocess_order = {"initial": 0, "outer_rim": 1, "background": 2}
metrics_table["preprocess_order"] = metrics_table["preprocess"].map(preprocess_order).fillna(99)
metrics_table = metrics_table.sort_values(["focus", "preprocess_order", "name"]).reset_index(drop=True)

output_csv = Path(__file__).resolve().parent / "metrics_summary.csv"
metrics_table.drop(columns=["preprocess_order"]).to_csv(output_csv, index=False)

for focus in [1, 2]:
    print(f"Focus {focus}")

    focus_table = (
        metrics_table[metrics_table["focus"] == focus]
        .drop(columns=["focus", "preprocess_order"])
        .reset_index(drop=True)
    )

    print(focus_table.to_string(index=False))
    print()