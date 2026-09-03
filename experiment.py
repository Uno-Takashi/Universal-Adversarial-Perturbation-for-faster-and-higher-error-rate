"""Run the universal-perturbation algorithm against one or more models.

Everything the experiment needs downloads itself: model weights from their own CDNs (or the
Hugging Face hub), and ILSVRC images through the Hugging Face dataset viewer. That makes a
run reproducible on a fresh machine with nothing but ``uv sync``.

Examples::

    # One model, 20 images
    uv run python experiment.py --models inception5h --num-images 20

    # Does the algorithm generalise past Inception?
    uv run python experiment.py --models inception5h mobilenet_v2 resnet50 --num-images 20

    # Sweep the multiplicity M (the `search_num` parameter of the paper)
    uv run python experiment.py --models mobilenet_v2 --search-num 1 3 5 10

    # Use a real local ILSVRC tree instead of Hugging Face
    uv run python experiment.py --dataset /datasets2/ILSVRC2012/train
"""

import argparse
import json
import os
import time

import numpy as np

from classifiers import AVAILABLE_MODELS, build_classifier
from imagenet_source import DEFAULT_DATASET, load_images
from universal_pert import universal_perturbation
from util_univ import fooling_rate_calc


def top1_accuracy(model, dataset, labels, batch_size):
    """Clean top-1 accuracy against ILSVRC ground truth, or None without labels.

    Predictions go through :meth:`classifiers.Classifier.to_ilsvrc` first, so a model with
    its own class ordering (Inception 5h) is scored correctly.
    """
    if labels.size == 0:
        return None
    predictions = np.concatenate(
        [model.predict(dataset[i : i + batch_size]) for i in range(0, len(dataset), batch_size)]
    )
    return float(np.mean(model.to_ilsvrc(predictions) == labels))


def evaluate(model, dataset, v, batch_size):
    """Fooling rate of ``v`` on ``model``, both unclipped and clipped to valid pixels."""
    return {
        "fooling_rate": fooling_rate_calc(v, dataset, model, batch_size=batch_size),
        "fooling_rate_clipped": fooling_rate_calc(
            v, dataset, model, batch_size=batch_size, clip=True
        ),
    }


def run_one(model_name, args, search_num):
    model = build_classifier(model_name)
    print(f"\n=== {model_name} (input {model.image_size}, {model.num_classes} classes) ===")

    dataset, labels = load_images(
        args.dataset,
        num_images=args.num_images,
        image_size=model.image_size,
        seed=args.seed,
    )
    print(f">> {len(dataset)} images loaded from {args.dataset}")

    accuracy = top1_accuracy(model, dataset, labels, args.batch_size)
    if accuracy is not None:
        print(f">> clean top-1 accuracy: {accuracy:.1%}")

    started = time.perf_counter()
    v = universal_perturbation(
        dataset,
        model,
        model.gradients,
        delta=args.delta,
        max_iter_uni=args.max_iter_uni,
        xi=args.xi,
        num_classes=args.num_classes,
        max_iter_df=args.max_iter_df,
        search_num=search_num,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started

    result = {
        "model": model_name,
        "search_num": search_num,
        "num_images": len(dataset),
        "xi": args.xi,
        "delta": args.delta,
        "image_size": list(model.image_size),
        "clean_top1": accuracy,
        "seconds": round(elapsed, 1),
        "perturbation_linf": float(np.max(np.abs(v))),
        **evaluate(model, dataset, v, args.batch_size),
    }

    print(
        f">> fooling rate {result['fooling_rate']:.1%} "
        f"(clipped {result['fooling_rate_clipped']:.1%}) "
        f"| l_inf {result['perturbation_linf']:.2f} | {elapsed:.0f}s"
    )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, f"universal_{model_name}_M{search_num}.npy"), v)

    return result, model, dataset, v


def run_transfer(entries, batch_size):
    """Evaluate each perturbation against every other model that shares its input size."""
    rows = []
    for source_name, (_, _, _, v) in entries.items():
        for target_name, (_, target_model, target_dataset, _) in entries.items():
            if source_name == target_name:
                continue
            if target_model.image_size != tuple(np.shape(v)[1:3]):
                continue
            rates = evaluate(target_model, target_dataset, v, batch_size)
            rows.append({"source": source_name, "target": target_name, **rates})
            print(
                f">> {source_name} -> {target_name}: "
                f"{rates['fooling_rate']:.1%} (clipped {rates['fooling_rate_clipped']:.1%})"
            )
    return rows


def print_table(results):
    header = (
        f"{'model':<18}{'M':>3}{'images':>8}{'clean':>8}{'fool':>8}{'fool(clip)':>12}{'sec':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        clean = f"{r['clean_top1']:.1%}" if r["clean_top1"] is not None else "-"
        print(
            f"{r['model']:<18}{r['search_num']:>3}{r['num_images']:>8}{clean:>8}"
            f"{r['fooling_rate']:>8.1%}{r['fooling_rate_clipped']:>12.1%}{r['seconds']:>7.0f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the universal-perturbation algorithm against one or more models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["inception5h"],
        help=f"models to attack. Available: {', '.join(AVAILABLE_MODELS)}",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id/shorthand, or a local ILSVRC-style directory",
    )
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument(
        "--search-num",
        nargs="+",
        type=int,
        default=[5],
        help="multiplicity M; pass several values to sweep it",
    )
    parser.add_argument("--delta", type=float, default=0.2, help="1 - target fooling rate")
    parser.add_argument("--xi", type=float, default=10.0, help="l_inf budget in pixel levels")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-iter-uni", type=int, default=5)
    parser.add_argument("--max-iter-df", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="also evaluate every perturbation against the other models",
    )
    parser.add_argument("--out", default=None, help="directory for perturbations and summary")
    return parser.parse_args()


def main():
    args = parse_args()

    results = []
    entries = {}
    for model_name in args.models:
        for search_num in args.search_num:
            result, model, dataset, v = run_one(model_name, args, search_num)
            results.append(result)
            if search_num == args.search_num[-1]:
                entries[model_name] = (result, model, dataset, v)

    print_table(results)

    transfer = []
    if args.transfer and len(entries) > 1:
        print("\n=== cross-model transfer ===")
        transfer = run_transfer(entries, args.batch_size)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        summary_path = os.path.join(args.out, "summary.json")
        with open(summary_path, "w") as fh:
            json.dump({"args": vars(args), "results": results, "transfer": transfer}, fh, indent=2)
        print(f"\n>> Summary written to {summary_path}")


if __name__ == "__main__":
    main()
