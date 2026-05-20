import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


PREPROCESS_BY_BACKBONE = {
    "resnet50v2": tf.keras.applications.resnet_v2.preprocess_input,
    "resnet101v2": tf.keras.applications.resnet_v2.preprocess_input,
    "resnet152v2": tf.keras.applications.resnet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "vgg19": tf.keras.applications.vgg19.preprocess_input,
    "efficientnetv2-b3": tf.keras.applications.efficientnet_v2.preprocess_input,
    "efficientnetv2-m": tf.keras.applications.efficientnet_v2.preprocess_input,
    "efficientnetv2-l": tf.keras.applications.efficientnet_v2.preprocess_input,
    "efficientnetv2s": tf.keras.applications.efficientnet_v2.preprocess_input,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

names = []

def pick_preprocess_from_model_name(model_name: str):
    name = model_name.lower()
    for key, fn in PREPROCESS_BY_BACKBONE.items():
        if key in name:
            names.append(key)
            return fn, key
    return lambda x: x, "none"


def parse_focuses_from_name(model_name: str, two_inputs: bool):
    tokens = model_name.split("_")
    numeric = [t for t in tokens if t.isdigit()]
    if two_inputs:
        if len(numeric) >= 2:
            return numeric[-2], numeric[-1]
        return "1", "2"
    if len(numeric) >= 1:
        return numeric[-1]
    return "1"


def load_image_array(path: Path, img_size):
    img = tf.keras.utils.load_img(str(path), target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    return arr.astype(np.float32)


def list_files(folder: Path):
    if not folder.exists():
        return {}
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.name] = p
    return out


def build_single_input_dataset(heldout_root: Path, image_type: str, focus: str, img_size):
    x, y = [], []

    good_dir = heldout_root / image_type / "good" / focus
    bad_dir = heldout_root / image_type / "bad" / focus
    bad_looks_good_dir = heldout_root / image_type / "bad_but_looks_good" / focus

    for p in good_dir.glob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            x.append(load_image_array(p, img_size))
            y.append(0)

    for source_dir in [bad_dir, bad_looks_good_dir]:
        for p in source_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                x.append(load_image_array(p, img_size))
                y.append(1)

    if len(x) == 0:
        raise ValueError("No images found for single-input heldout dataset.")

    return np.stack(x), np.array(y, dtype=np.int32)


def build_two_input_pairs_for_class(base_dir: Path, class_name: str, focus_a: str, focus_b: str):
    dir_a = base_dir / class_name / focus_a
    dir_b = base_dir / class_name / focus_b

    files_a = list_files(dir_a)
    files_b = list_files(dir_b)

    common = sorted(set(files_a.keys()) & set(files_b.keys()))
    pairs = [(files_a[f], files_b[f]) for f in common]
    return pairs


def build_two_input_dataset(heldout_root: Path, image_type: str, focus_a: str, focus_b: str, img_size):
    base = heldout_root / image_type

    good_pairs = build_two_input_pairs_for_class(base, "good", focus_a, focus_b)
    bad_pairs = build_two_input_pairs_for_class(base, "bad", focus_a, focus_b)
    bad_looks_good_pairs = build_two_input_pairs_for_class(base, "bad_but_looks_good", focus_a, focus_b)

    all_pairs = [(p1, p2, 0) for p1, p2 in good_pairs]
    all_pairs += [(p1, p2, 1) for p1, p2 in bad_pairs]
    all_pairs += [(p1, p2, 1) for p1, p2 in bad_looks_good_pairs]

    if len(all_pairs) == 0:
        raise ValueError("No paired images found for two-input heldout dataset.")

    x1, x2, y = [], [], []
    for p1, p2, label in all_pairs:
        x1.append(load_image_array(p1, img_size))
        x2.append(load_image_array(p2, img_size))
        y.append(label)

    return np.stack(x1), np.stack(x2), np.array(y, dtype=np.int32)


def build_rulebased_dataset(
    heldout_root: Path,
    image_type: str,
    focus_a: str,
    focus_b: str,
    img_size,
):
    
    base = heldout_root / image_type

    class_label = {"good": 0, "bad": 1, "bad_but_looks_good": 1}

    all_entries = []
    for class_name, label in class_label.items():
        pairs = build_two_input_pairs_for_class(base, class_name, focus_a, focus_b)
        all_entries.extend((p_a, p_b, label) for p_a, p_b in pairs)

    if len(all_entries) == 0:
        raise ValueError(
            f"No paired images found for ensemble dataset "
            f"(focus_a={focus_a}, focus_b={focus_b})."
        )

    x_a, x_b, y = [], [], []
    for p_a, p_b, label in all_entries:
        x_a.append(load_image_array(p_a, img_size))
        x_b.append(load_image_array(p_b, img_size))
        y.append(label)

    return np.stack(x_a), np.stack(x_b), np.array(y, dtype=np.int32)


def save_confusion_matrix_plot(cm, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["good(0)", "bad(1)"]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_roc_plot(y_true, y_score, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_one_model(model_path: Path, heldout_root: Path, image_type: str, img_size, threshold: float, batch_size: int):
    model_name = model_path.stem
    model = tf.keras.models.load_model(str(model_path), compile=False)
    two_inputs = len(model.inputs) == 2

    preprocess_fn, preprocess_name = pick_preprocess_from_model_name(model_name)

    if two_inputs:
        focus_a, focus_b = parse_focuses_from_name(model_name, two_inputs=True)
        x1, x2, y_true = build_two_input_dataset(heldout_root, image_type, focus_a, focus_b, img_size)
        x1 = preprocess_fn(x1)
        x2 = preprocess_fn(x2)
        y_score = model.predict([x1, x2], batch_size=batch_size, verbose=0).reshape(-1)
    else:
        focus = parse_focuses_from_name(model_name, two_inputs=False)
        x, y_true = build_single_input_dataset(heldout_root, image_type, focus, img_size)
        x = preprocess_fn(x)
        y_score = model.predict(x, batch_size=batch_size, verbose=0).reshape(-1)

    y_pred = (y_score >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "model_file": model_path.name,
        "model_name": model_name,
        "two_inputs": bool(two_inputs),
        "preprocess": preprocess_name,
        "image_type": image_type,
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    unique = np.unique(y_true)
    if len(unique) == 2:
        result["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["auc"] = np.nan

    return result, y_true, y_score, cm


def evaluate_rulebased_models(
    model_path_a: Path,
    model_path_b: Path,
    focus_a: str,
    focus_b: str,
    heldout_root: Path,
    image_type: str,
    img_size,
    threshold: float,
    batch_size: int,
):

    model_a = tf.keras.models.load_model(str(model_path_a), compile=False)
    model_b = tf.keras.models.load_model(str(model_path_b), compile=False)

    preprocess_fn_a, preprocess_name_a = pick_preprocess_from_model_name(model_path_a.stem)
    preprocess_fn_b, preprocess_name_b = pick_preprocess_from_model_name(model_path_b.stem)


    x_a_raw, x_b_raw, y_true = build_rulebased_dataset(
        heldout_root, image_type, focus_a, focus_b, img_size
    )


    x_a = preprocess_fn_a(x_a_raw.copy())
    x_b = preprocess_fn_b(x_b_raw.copy())

    score_a = model_a.predict(x_a, batch_size=batch_size, verbose=0).reshape(-1)
    score_b = model_b.predict(x_b, batch_size=batch_size, verbose=0).reshape(-1)

    pred_a = (score_a >= threshold).astype(np.int32)
    pred_b = (score_b >= threshold).astype(np.int32)

    y_pred = np.where((pred_a == 0) & (pred_b == 0), 0, 1).astype(np.int32)


    y_score = np.maximum(score_a, score_b)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    ensemble_name = f"rule_based_{names[0]}_1_2"

    result = {
        "model_file": f"{model_path_a.name} + {model_path_b.name}",
        "model_name": ensemble_name,
        "two_inputs": False,
        "preprocess": f"{preprocess_name_a} / {preprocess_name_b}",
        "image_type": image_type,
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    unique = np.unique(y_true)
    if len(unique) == 2:
        result["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["auc"] = np.nan

    return result, y_true, y_score, cm, ensemble_name


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved models on heldout dataset.")
    parser.add_argument("--models_dir", type=str, default="../model_training/saved_models")
    parser.add_argument("--heldout_root", type=str, default="../dataset_creation/processed_heldout")
    parser.add_argument("--image_type", type=str, default="initial", choices=["initial"])
    parser.add_argument("--img_h", type=int, default=300)
    parser.add_argument("--img_w", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./heldout_eval")

    
    parser.add_argument(
        "--rulebased", nargs=2, metavar=("MODEL_A", "MODEL_B"), default=None,
        help=("example --rulebased model_focus1.keras model_focus2.keras"),)
    parser.add_argument(
        "--rulebased_focuses", nargs=2, metavar=("FOCUS_A", "FOCUS_B"), 
        default=("1", "2"), help=("example --rulebased_focuses 1 2"),) # make sure model and focus folder are in same order!

    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_h, args.img_w)
    results = []


    if args.rulebased is not None:
        if args.rulebased_focuses is None:
            parser.error("--rulebased_focuses FOCUS_A FOCUS_B is required when --rulebased is used.")

        def resolve_model(name_or_path: str) -> Path:
            p = Path(name_or_path)
            if p.is_absolute() and p.exists():
                return p
            candidate = models_dir / name_or_path
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Cannot find model: {name_or_path}")

        model_path_a = resolve_model(args.rulebased[0])
        model_path_b = resolve_model(args.rulebased[1])
        focus_a, focus_b = args.rulebased_focuses[0], args.rulebased_focuses[1]

        print(f"[RULE-BASED] {model_path_a.name} (focus={focus_a})  AND  {model_path_b.name} (focus={focus_b})")

        try:
            result, y_true, y_score, cm, ensemble_name = evaluate_rulebased_models(
                model_path_a=model_path_a,
                model_path_b=model_path_b,
                focus_a=focus_a,
                focus_b=focus_b,
                heldout_root=heldout_root,
                image_type=args.image_type,
                img_size=img_size,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
            results.append(result)

            save_confusion_matrix_plot(
                cm,
                output_dir / f"{ensemble_name}_cm.png",
                title=f"Confusion Matrix - {ensemble_name}",
            )
            if len(np.unique(y_true)) == 2:
                save_roc_plot(
                    y_true,
                    y_score,
                    output_dir / f"{ensemble_name}_roc.png",
                    title=f"ROC - {ensemble_name}",
                )

            print(f"[OK] Ensemble evaluated. AUC={result.get('auc', float('nan')):.4f}  "
                  f"F1={result['f1']:.4f}  Acc={result['accuracy']:.4f}")

        except Exception as e:
            print(f"[FAIL] Ensemble: {e}")
            raise

    
    else:
        model_files = sorted(
            [p for p in models_dir.glob("*.h5")] + [p for p in models_dir.glob("*.keras")]
        )

        if not model_files:
            raise FileNotFoundError(f"No model files found in: {models_dir}")

        for model_path in model_files:
            try:
                result, y_true, y_score, cm = evaluate_one_model(
                    model_path=model_path,
                    heldout_root=heldout_root,
                    image_type=args.image_type,
                    img_size=img_size,
                    threshold=args.threshold,
                    batch_size=args.batch_size,
                )
                results.append(result)

                base_name = model_path.stem
                save_confusion_matrix_plot(
                    cm,
                    output_dir / f"{base_name}_cm.png",
                    title=f"Confusion Matrix - {base_name}",
                )

                if len(np.unique(y_true)) == 2:
                    save_roc_plot(
                        y_true,
                        y_score,
                        output_dir / f"{base_name}_roc.png",
                        title=f"ROC - {base_name}",
                    )

                print(f"[OK] {model_path.name} evaluated.")

            except Exception as e:
                print(f"[FAIL] {model_path.name}: {e}")

    if len(results) == 0:
        raise RuntimeError("No models were evaluated successfully.")

    df = pd.DataFrame(results).sort_values(by="auc", ascending=False, na_position="last")
    out_csv = output_dir / "heldout_metrics_summary_production_resnetv2.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(f"- {out_csv}")
    print(f"- Confusion matrix and ROC plots in: {output_dir}")


if __name__ == "__main__":
    main()