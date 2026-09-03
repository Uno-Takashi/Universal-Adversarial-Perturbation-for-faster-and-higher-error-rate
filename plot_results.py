"""Turn experiment summaries into the figures and tables the README carries.

Reads the ``summary.json`` files written by ``experiment.py --out`` and produces, for both a
light and a dark surface, three figures:

* ``fooling_vs_images``  -- held-out fooling rate against generation-set size, with each
  model's random-perturbation baseline as a dashed line of the same hue. A perturbation is
  universal only where the solid line is above its dashed one.
* ``margin_vs_images``   -- the same thing as a single number per model, ``val - random``,
  against a zero reference. Above zero is the whole claim.
* ``fooling_vs_multiplicity`` -- held-out fooling rate against the multiplicity M
  (``search_num``), for a fixed generation-set size.

Usage::

    uv run python plot_results.py results/ --out docs
"""

import argparse
import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots from a validated palette: light and dark are separately stepped for their
# own surface, not an automatic flip of one another.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#dedcd7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"],
    },
}

MODEL_ORDER = ["inception5h", "mobilenet_v2", "resnet50", "inception_v3", "vgg16"]


def load_results(root):
    """Return ``{model: [result, ...]}`` from every summary.json under ``root``."""
    results = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        with open(path) as handle:
            payload = json.load(handle)
        for entry in payload.get("results", []):
            results[entry["model"]].append(entry)
    return results


def _ordered_models(results):
    known = [m for m in MODEL_ORDER if m in results]
    return known + sorted(m for m in results if m not in MODEL_ORDER)


def _style_axes(ax, theme, xlabel, ylabel, title):
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    ax.set_title(title, color=theme["text"], fontsize=13, pad=14, loc="left")
    ax.set_xlabel(xlabel, color=theme["muted"], fontsize=10)
    ax.set_ylabel(ylabel, color=theme["muted"], fontsize=10)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _series_points(entries, x_key, y_key):
    points = sorted((e[x_key], e[y_key]) for e in entries if e.get(y_key) is not None)
    return [p[0] for p in points], [p[1] * 100 for p in points]


def plot_fooling_vs_images(results, theme, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    models = _ordered_models(results)

    for index, model in enumerate(models):
        colour = theme["series"][index % len(theme["series"])]
        entries = [e for e in results[model] if e.get("num_val_images")]
        x, y = _series_points(entries, "num_images", "val_fooling_rate_clipped")
        xb, yb = _series_points(entries, "num_images", "val_random_baseline_clipped")
        if not x:
            continue
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6, label=model, zorder=3)
        ax.plot(xb, yb, color=colour, linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.75, zorder=2)
        # Direct label: identity must not depend on colour alone.
        ax.annotate(
            model,
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            color=colour,
            fontsize=9,
            va="center",
        )

    _style_axes(
        ax,
        theme,
        "images the perturbation was fitted to",
        "fooling rate on held-out images (%)",
        "Solid: universal perturbation.  Dashed: random perturbation, same $l_\\infty$ budget.",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(left=13)
    ax.margins(x=0.16)
    ax.legend(frameon=False, labelcolor=theme["muted"], fontsize=9, loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def plot_margin_vs_images(results, theme, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    models = _ordered_models(results)

    for index, model in enumerate(models):
        colour = theme["series"][index % len(theme["series"])]
        entries = [e for e in results[model] if e.get("num_val_images")]
        points = sorted(
            (
                e["num_images"],
                (e["val_fooling_rate_clipped"] - e["val_random_baseline_clipped"]) * 100,
            )
            for e in entries
        )
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6, label=model, zorder=3)
        ax.annotate(
            model,
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            color=colour,
            fontsize=9,
            va="center",
        )

    ax.axhline(0, color=theme["text"], linewidth=1.2, alpha=0.55, zorder=1)
    ax.annotate(
        "at or below zero: no better than random noise",
        (1.0, 0.0),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        color=theme["muted"],
        fontsize=9,
    )
    _style_axes(
        ax,
        theme,
        "images the perturbation was fitted to",
        "held-out fooling rate minus random baseline (pt)",
        "How many images does a universal perturbation need?",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 128, 256, 512, 1024])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(left=13)
    ax.margins(x=0.24)  # room for the direct labels past the last point
    ax.legend(frameon=False, labelcolor=theme["muted"], fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def plot_fooling_vs_multiplicity(results, theme, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    models = _ordered_models(results)
    plotted = False

    for index, model in enumerate(models):
        colour = theme["series"][index % len(theme["series"])]
        entries = results[model]
        x, y = _series_points(entries, "search_num", "val_fooling_rate_clipped")
        # Only a real sweep of M is worth a chart; a single value repeated across
        # generation-set sizes is not.
        if len(set(x)) < 2:
            continue
        plotted = True
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6, label=model, zorder=3)
        xb, yb = _series_points(entries, "search_num", "val_random_baseline_clipped")
        if xb:
            ax.plot(
                xb, yb, color=colour, linewidth=1.4, linestyle=(0, (4, 3)), alpha=0.75, zorder=2
            )
        ax.annotate(
            model,
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            color=colour,
            fontsize=9,
            va="center",
        )

    if not plotted:
        plt.close(fig)
        return False

    sizes = sorted({e["num_images"] for m in results for e in results[m]})
    subtitle = f"generation set: {sizes[0]} images" if len(sizes) == 1 else ""
    _style_axes(
        ax,
        theme,
        "multiplicity M (targets attacked per image, `search_num`)",
        "fooling rate on held-out images (%)",
        f"Does attacking more classes per image help?  {subtitle}",
    )
    ax.set_xticks(range(0, 21, 2))
    ax.margins(x=0.14)
    ax.legend(frameon=False, labelcolor=theme["muted"], fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return True


def markdown_tables(results):
    """Render one table per model, plus a cross-model margin table."""
    lines = []
    models = _ordered_models(results)

    lines.append("| gen images | " + " | ".join(models) + " |")
    lines.append("|---|" + "---|" * len(models))
    sizes = sorted({e["num_images"] for m in models for e in results[m]})
    for size in sizes:
        cells = []
        for model in models:
            match = [e for e in results[model] if e["num_images"] == size]
            if match:
                entry = match[-1]
                margin = (
                    entry["val_fooling_rate_clipped"] - entry["val_random_baseline_clipped"]
                ) * 100
                cells.append(f"{margin:+.1f}pt")
            else:
                cells.append("-")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")

    for model in models:
        lines.append(f"\n### {model}\n")
        lines.append("| gen images | gen fool | val fool | random | margin | clean top-1 | sec |")
        lines.append("|---|---|---|---|---|---|---|")
        for entry in sorted(results[model], key=lambda e: (e["num_images"], e["search_num"])):
            margin = (
                entry["val_fooling_rate_clipped"] - entry["val_random_baseline_clipped"]
            ) * 100
            clean = (
                f"{entry['clean_top1_val']:.1%}" if entry.get("clean_top1_val") is not None else "-"
            )
            lines.append(
                f"| {entry['num_images']} | {entry['train_fooling_rate_clipped']:.1%} | "
                f"**{entry['val_fooling_rate_clipped']:.1%}** | "
                f"{entry['val_random_baseline_clipped']:.1%} | {margin:+.1f}pt | {clean} | "
                f"{entry['seconds']:.0f} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="directory containing summary.json files")
    parser.add_argument("--out", default="docs", help="where to write the figures")
    parser.add_argument("--prefix", default="", help="filename prefix for the figures")
    args = parser.parse_args()

    results = load_results(args.results)
    if not results:
        raise SystemExit(f"No summary.json found under {args.results!r}")

    os.makedirs(args.out, exist_ok=True)
    for mode, theme in THEMES.items():
        for name, fn in [
            ("fooling_vs_images", plot_fooling_vs_images),
            ("margin_vs_images", plot_margin_vs_images),
            ("fooling_vs_multiplicity", plot_fooling_vs_multiplicity),
        ]:
            path = os.path.join(args.out, f"{args.prefix}{name}-{mode}.png")
            if fn(results, theme, path) is not False:
                print(f"wrote {path}")

    print()
    print(markdown_tables(results))


if __name__ == "__main__":
    main()
