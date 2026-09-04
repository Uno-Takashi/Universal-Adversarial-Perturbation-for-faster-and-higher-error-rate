"""Collect experiment summaries into the JSON the documentation site renders.

``experiment.py --out DIR`` leaves one ``summary.json`` per run. This walks any number of
those trees, tags each run with the sweep it belongs to, and writes a single file the MDX
pages import. Running it again after more results land refreshes the site without touching
any prose.

Usage::

    uv run python export_docs_data.py \\
        --sweep images=results/nsweep \\
        --sweep multiplicity=results/msweep \\
        --sweep modern=results/modern \\
        --out docs/data/results.json
"""

import argparse
import datetime
import glob
import json
import os

# Paradigm labels so the site can group models by architecture family rather than by name.
PARADIGMS = {
    "inception5h": "CNN (2014, the paper's model)",
    "inception_v3": "CNN (2015)",
    "resnet50": "CNN (2015)",
    "vgg16": "CNN (2014)",
    "mobilenet_v2": "efficient CNN (2018)",
    "mobilenet_v3_large": "efficient CNN (2019)",
    "efficientnet_b0": "efficient CNN (2019)",
    "convnext_tiny": "modernised CNN (2022)",
    "resnet_vd_50_ssld": "distillation-trained CNN",
    "vit_b16": "vision transformer (2020)",
    "swin_tiny": "hierarchical transformer (2021)",
    "swin_base": "hierarchical transformer (2021)",
    "deit_b16_distilled": "distillation-trained transformer (2021)",
}


def load_sweep(root, label):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        with open(path) as handle:
            payload = json.load(handle)
        for entry in payload.get("results", []):
            val = entry["val_fooling_rate_clipped"]
            baseline = entry["val_random_baseline_clipped"]
            runs.append(
                {
                    "sweep": label,
                    "model": entry["model"],
                    "paradigm": PARADIGMS.get(entry["model"], "unknown"),
                    "numImages": entry["num_images"],
                    "numValImages": entry.get("num_val_images"),
                    "searchNum": entry["search_num"],
                    "imageSize": entry.get("image_size"),
                    "xi": entry.get("xi"),
                    "genFooling": entry["train_fooling_rate_clipped"],
                    "valFooling": val,
                    "randomBaseline": baseline,
                    "margin": val - baseline,
                    "cleanTop1": entry.get("clean_top1_val"),
                    "seconds": entry.get("seconds"),
                    "perturbationLinf": entry.get("perturbation_linf"),
                }
            )
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="a results tree to include, tagged with a label; repeatable",
    )
    parser.add_argument("--out", default=os.path.join("docs", "data", "results.json"))
    args = parser.parse_args()

    runs = []
    counts = {}
    for spec in args.sweep:
        if "=" not in spec:
            raise SystemExit(f"--sweep expects LABEL=DIR, got {spec!r}")
        label, _, root = spec.partition("=")
        found = load_sweep(root, label) if os.path.isdir(root) else []
        counts[label] = len(found)
        runs.extend(found)

    payload = {
        "generatedAt": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "runCount": len(runs),
        "sweepCounts": counts,
        "models": sorted({run["model"] for run in runs}),
        "runs": runs,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"wrote {args.out}: {len(runs)} runs {counts}")


if __name__ == "__main__":
    main()
