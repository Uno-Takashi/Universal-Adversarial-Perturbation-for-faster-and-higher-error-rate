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


def random_baseline(model, dataset, shape, xi, batch_size, seed=0, repeats=3):
    """Fooling rate of a random sign perturbation at the same l_inf budget.

    Without this, a held-out fooling rate is uninterpretable: some fraction of predictions
    flips under *any* perturbation of this magnitude. A universal perturbation is only
    interesting to the extent it beats this number. The signs are saturated to +/-xi because
    that is the structure the projected algorithm produces.
    """
    rng = np.random.default_rng(seed)
    rates = [
        fooling_rate_calc(
            rng.choice([-xi, xi], size=shape).astype(np.float32),
            dataset,
            model,
            batch_size=batch_size,
            clip=True,
        )
        for _ in range(repeats)
    ]
    return float(np.mean(rates))


def run_one(model_name, args, search_num):
    model = build_classifier(model_name)
    print(f"\n=== {model_name} (input {model.image_size}, {model.num_classes} classes) ===")

    # One draw, split disjointly: the perturbation is fitted on `gen` and scored on `val`.
    # Evaluating on `gen` alone would just report the algorithm's own stopping criterion.
    images, labels = load_images(
        args.dataset,
        num_images=args.num_images + args.num_val_images,
        image_size=model.image_size,
        seed=args.seed,
    )
    if len(images) < args.num_images + args.num_val_images:
        raise ValueError(
            f"asked for {args.num_images} + {args.num_val_images} images, got {len(images)}"
        )
    gen = np.array(images[: args.num_images])
    val = np.array(images[args.num_images :])
    val_labels = labels[args.num_images :] if labels.size else labels
    print(f">> {len(gen)} generation + {len(val)} validation images from {args.dataset}")

    accuracy = top1_accuracy(model, val, val_labels, args.batch_size)
    if accuracy is not None:
        print(f">> clean top-1 accuracy on validation: {accuracy:.1%}")

    started = time.perf_counter()
    v = universal_perturbation(
        gen,
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

    train_rates = evaluate(model, gen, v, args.batch_size)
    val_rates = evaluate(model, val, v, args.batch_size)
    baseline = random_baseline(model, val, np.shape(v), args.xi, args.batch_size, seed=args.seed)

    result = {
        "model": model_name,
        "search_num": search_num,
        "num_images": len(gen),
        "num_val_images": len(val),
        "xi": args.xi,
        "delta": args.delta,
        "image_size": list(model.image_size),
        "clean_top1_val": accuracy,
        "seconds": round(elapsed, 1),
        "perturbation_linf": float(np.max(np.abs(v))),
        "train_fooling_rate": train_rates["fooling_rate"],
        "train_fooling_rate_clipped": train_rates["fooling_rate_clipped"],
        "val_fooling_rate": val_rates["fooling_rate"],
        "val_fooling_rate_clipped": val_rates["fooling_rate_clipped"],
        "val_random_baseline_clipped": baseline,
    }

    print(
        f">> generation set {train_rates['fooling_rate_clipped']:.1%} "
        f"| validation {val_rates['fooling_rate_clipped']:.1%} "
        f"| random baseline {baseline:.1%} "
        f"| l_inf {result['perturbation_linf']:.2f} | {elapsed:.0f}s"
    )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, f"universal_{model_name}_M{search_num}.npy"), v)

    return result, model, val, v


def run_transfer(entries, batch_size):
    """Evaluate each perturbation on the other models, using their held-out images."""
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
    columns = (
        f"{'model':<18}{'M':>3}{'gen':>5}{'val':>5}{'clean':>8}"
        f"{'gen fool':>10}{'val fool':>10}{'random':>8}{'sec':>7}"
    )
    print("\n" + columns)
    print("-" * len(columns))
    for r in results:
        clean = f"{r['clean_top1_val']:.1%}" if r["clean_top1_val"] is not None else "-"
        print(
            f"{r['model']:<18}{r['search_num']:>3}{r['num_images']:>5}{r['num_val_images']:>5}"
            f"{clean:>8}{r['train_fooling_rate_clipped']:>10.1%}"
            f"{r['val_fooling_rate_clipped']:>10.1%}"
            f"{r['val_random_baseline_clipped']:>8.1%}{r['seconds']:>7.0f}"
        )
    print(
        "\n'gen fool' is measured on the images the perturbation was fitted to, so it "
        "restates\nthe algorithm's own stopping criterion. Only 'val fool' minus 'random' "
        "says whether\nthe perturbation is universal."
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
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="images the perturbation is fitted to; below ~100 it overfits and does not generalise",
    )
    parser.add_argument(
        "--num-val-images",
        type=int,
        default=200,
        help="held-out images the perturbation is scored on (disjoint from --num-images)",
    )
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
            result, model, val, v = run_one(model_name, args, search_num)
            results.append(result)
            if search_num == args.search_num[-1]:
                # Transfer is scored on held-out images too.
                entries[model_name] = (result, model, val, v)

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
