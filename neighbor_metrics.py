import argparse
import math
import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import csv

#---
#NOTE: This script computes neighborhood metrics from extracted ViG KNN graph files.
#It is meant to be run through main.py using a YAML config with task: neighbor_metrics
#and run_mode: metrics.
#
#It expects graph files created by extract_graph.py. Each graph file should contain
#a knn_output tensor with shape (2, B, N, k), where N is the number of graph nodes
#and k is the number of neighbors.
#
#The script supports comparing multiple patch/grid runs, such as P14, P56, and P112.
#The YAML controls which graph directories are included and which blocks are measured.
#Each graph directory is expected to contain files named block01.pt, block02.pt, etc.
#
#Metrics computed:
# 1. Node-neighbor label correctness:
#    Measures how often each node connects to neighbors with the same ice/ocean label.
#
# 2. Mutual neighbor rate:
#    Measures how often directed neighbor pairs are reciprocal.
#    Example: node i has neighbor j, and node j also has neighbor i.
#
# 3. Mean neighbor distance:
#    Measures average Euclidean distance between each node and its neighbors in grid units.
#    A normalized version is also saved so different grid sizes can be compared.
#
# 4. Most popular neighbors:
#    Reports nodes that appear most often as neighbors across the graph.
#
#Outputs:
#  - neighbor_metrics_summary.csv
#  - grouped bar plots comparing patch/grid runs across blocks
#  - optional individual plots per patch/grid run
#
#YAML options:
#  make_grouped_plots: true/false controls whether grouped comparison plots are saved.
#  make_individual_plots: true/false controls whether older per-patch plots are saved.
#
#Command line options:
#  --skip_grouped_plots skips the grouped comparison figures.
#  --make_individual_plots saves the older per-patch line/bar plots.
#---


"""
Grab the block number from the title given 
"""
def block_num_from_title(title: str) -> int:
    b = title.split()[-1].replace("B", "")
    return int(b)


"""
For the grouped bar charts, normalized the avg distance so that it's a percent of the grid size for better comparison
"""
def normalize_distance(mean_dist_grid_units: float, grid_h: int, grid_w: int) -> float:
    if not np.isfinite(mean_dist_grid_units):
        return float("nan")
    max_dist = math.sqrt((grid_w - 1) ** 2 + (grid_h - 1) ** 2)
    if max_dist <= 0:
        return float("nan")
    return 100.0 * (mean_dist_grid_units / max_dist)


"""
Adds number annotations to each bar in the grouped bar charts
"""
def annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False
        )


"""
Creates a grouped bar chart across blocks for the three metrics across blocks 1-6, 12 
"""
def plot_grouped_bars_across_patches(
    results_by_patch: dict,
    blocks: list[int],
    metric_key: str,
    ylabel: str,
    title: str,
    outpath: str | None = None,
    show: bool = False,
    colors: dict | None = None,
    value_labels: bool = True,
    ylim: tuple[float, float] | None = None,
    style: dict | None = None,
    legend_outside: bool = True,   # normal plot: keep legend inside unless you want otherwise
):
    default_style = {
        "ticks": 14,
        "labels": 16,
        "title": 18,
        "legend": 14,
        "annot": 12,
        "grid_lw": 0.8,
        "grid_alpha": 0.30,
    }

    if style is None:
        style = default_style
    else:
        tmp = default_style.copy()
        tmp.update(style)
        style = tmp

    patch_order = list(results_by_patch.keys())
    colors = colors or {}

    # Collect values per patch per block
    vals = {}
    for patch, res_list in results_by_patch.items():
        block_map = {}
        for d in res_list:
            b = block_num_from_title(d["title"])
            block_map[b] = d.get(metric_key, float("nan"))
        vals[patch] = block_map

    x = np.arange(len(blocks))
    width = 0.8 / max(len(patch_order), 1)

    # NORMAL figure size + ONE layout system
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    for i, patch in enumerate(patch_order):
        y = [vals[patch].get(b, float("nan")) for b in blocks]
        bars = ax.bar(
            x + (i - (len(patch_order) - 1) / 2) * width,
            y,
            width=width,
            label=patch,
            color=colors.get(patch, None),
            edgecolor="black",
            linewidth=0.6,
        )
        if value_labels:
            annotate_bars(ax, bars, fmt="{:.1f}", fontsize=style["annot"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b:02d}" for b in blocks], fontsize=style["ticks"])
    ax.tick_params(axis="y", labelsize=style["ticks"])

    ax.set_xlabel("Block", fontsize=style["labels"])
    ax.set_ylabel(ylabel, fontsize=style["labels"])
    ax.set_title(title, fontsize=style["title"])

    ax.grid(True, axis="y", linewidth=style["grid_lw"], alpha=style["grid_alpha"])
    ax.set_axisbelow(True)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend_outside:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(patch_order),
            frameon=False,
            fontsize=style["legend"],
        )
    else:
        ax.legend(frameon=False, fontsize=style["legend"])

    # Save / show — NO bbox_inches="tight" for normal behavior
    if outpath:
        fig.savefig(outpath + ".png")
        fig.savefig(outpath + ".svg")
    if show:
        plt.show()

    plt.close(fig)


"""
From the H/W in run one, get and store the normalized distance 
"""
def add_normalized_distance(results_list: list[dict]):
    for d in results_list:
        H = d["grid_h"]
        W = d["grid_w"]
        d["mean_neighbor_dist_norm_pct"] = normalize_distance(
            d["mean_neighbor_dist_overall"], H, W
        )
        d["mean_neighbor_dist_ice_norm_pct"] = normalize_distance(
            d["mean_neighbor_dist_ice"], H, W
        )
        d["mean_neighbor_dist_ocean_norm_pct"] = normalize_distance(
            d["mean_neighbor_dist_ocean"], H, W
        )


"""
Used to get the block number from the title for the plots
"""
def _block_label_from_title(title: str) -> str:
    parts = title.split()
    return parts[-1] if parts else title


"""
Function to plot the mean correctness of node and neighbor label agreements 
"""
def plot_means_across_blocks(results_for_patch: list[dict], patch_name: str, outpath: str | None = None, show: bool = False):
    def block_key(d):
        b = _block_label_from_title(d["title"]).replace("B", "")
        return int(b) if b.isdigit() else 999

    results_for_patch = sorted(results_for_patch, key=block_key)
    x_labels = [_block_label_from_title(d["title"]) for d in results_for_patch]
    x = np.arange(len(x_labels))

    overall = [d["mean_overall_pct"] for d in results_for_patch]
    ice = [d["mean_ice_pct"] for d in results_for_patch]
    ocean = [d["mean_ocean_pct"] for d in results_for_patch]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, overall, marker="o", label="Overall mean", color='black')
    plt.plot(x, ice, marker="o", label="Ice mean", color='blue')
    plt.plot(x, ocean, marker="o", label="Ocean mean", color="green")
    plt.xticks(x, x_labels)
    plt.ylim(0, 100)
    plt.xlabel("Block")
    plt.ylabel("Mean % neighbors matching center label")
    plt.title(f"{patch_name}: Node-Neighbor Label Correctness Rates Across Blocks")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend()

    #save or show or both
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


"""
Function to plot the mutual neighbor rate ie center i has neighbor j, center j also has neighbor i 
"""
def plot_mutual_rate_across_blocks(results_for_patch: list[dict], patch_name: str, outpath: str | None = None, show: bool = False):
    def block_key(d):
        b = _block_label_from_title(d["title"]).replace("B", "")
        return int(b) if b.isdigit() else 999

    results_for_patch = sorted(results_for_patch, key=block_key)
    x_labels = [_block_label_from_title(d["title"]) for d in results_for_patch]
    x = np.arange(len(x_labels))

    mutual = [d["mutual_rate_pct"] for d in results_for_patch]

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, mutual, color='blue')
    plt.xticks(x, x_labels)
    plt.ylim(0, 100)
    plt.xlabel("Block")
    plt.ylabel("Mutual neighbor rate (%)")
    plt.title(f"{patch_name}: Mutual Neighbor Rate Across Blocks")
    plt.grid(True, axis="y", linewidth=0.5, alpha=0.5)

    #save or show or both
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


"""
Function to get the 2D coord for each node to calc euclidean distance 
"""
def node_xy_from_grid(grid_h: int, grid_w: int) -> np.ndarray:
    rows = np.repeat(np.arange(grid_h), grid_w)
    cols = np.tile(np.arange(grid_w), grid_h)
    return np.stack([cols, rows], axis=1).astype(np.float64)


"""
Function to calculate the avg euclidean distance for each node and it's neighbors 
"""
def mean_neighbor_distances_from_idx(idx: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    idx_np = idx.cpu().numpy()
    N, k = idx_np.shape

    xy = node_xy_from_grid(grid_h, grid_w)
    dmean = np.full(N, np.nan, dtype=np.float64)

    for i in range(N):
        nbrs = idx_np[i]
        nbrs = nbrs[(nbrs >= 0) & (nbrs < N)]
        nbrs = nbrs[nbrs != i]
        if nbrs.size == 0:
            continue

        dx = xy[nbrs, 0] - xy[i, 0]
        dy = xy[nbrs, 1] - xy[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        dmean[i] = float(np.mean(d))

    return dmean


"""
Function to plot the avg euclidean distance overall and for ice and ocean nodes
"""
def plot_neighbor_distance_across_blocks(results_for_patch: list[dict], patch_name: str, outpath: str | None = None, show: bool = False):
    def block_key(d):
        b = _block_label_from_title(d["title"]).replace("B", "")
        return int(b) if b.isdigit() else 999

    results_for_patch = sorted(results_for_patch, key=block_key)
    x_labels = [_block_label_from_title(d["title"]) for d in results_for_patch]
    x = np.arange(len(x_labels))

    overall = [d["mean_neighbor_dist_overall"] for d in results_for_patch]
    ice = [d["mean_neighbor_dist_ice"] for d in results_for_patch]
    ocean = [d["mean_neighbor_dist_ocean"] for d in results_for_patch]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, overall, marker="o", linewidth=2, color="black", label="Overall mean")
    plt.plot(x, ice, marker="o", linewidth=2, color="blue", label="Ice mean")
    plt.plot(x, ocean, marker="o", linewidth=2, color="green", label="Ocean mean")
    plt.xticks(x, x_labels)
    plt.xlabel("Block")
    plt.ylabel("Mean neighbor distance in grid units")
    plt.title(f"{patch_name}: Mean neighbor distance across blocks")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend()

    #save or show or both
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()


"""
This function figures out which node appears most as a neighbor across the entire graph. Centers are included 
on the neighbor list, so I have excluded the center as a neighbor for these metrics.
"""
def most_popular_neighbors(idx: torch.Tensor, topk: int = 10):
    idx_np = idx.cpu().numpy()
    N, k = idx_np.shape

    centers = np.repeat(np.arange(N), k)
    neighbors = idx_np.reshape(-1)

    valid = (neighbors >= 0) & (neighbors < N)
    centers = centers[valid]
    neighbors = neighbors[valid]

    mask_no_self = centers != neighbors
    neighbors_no_self = neighbors[mask_no_self]

    counts = (
        np.bincount(neighbors_no_self, minlength=N)
        if neighbors_no_self.size > 0
        else np.zeros(N, dtype=np.int64)
    )

    top = np.argsort(counts)[::-1][:topk]

    return {
        "counts": counts,
        "top": top,
        "total_slots": int(neighbors_no_self.size),
    }


"""
A function to figure out the rate of reciprical directed neighbor relationships in the graph 
"""
def mutual_neighbor_rate(idx: torch.Tensor, exclude_self: bool = True) -> float:
    idx_np = idx.cpu().numpy()
    N, k = idx_np.shape

    edges = set()
    for i in range(N):
        for j in idx_np[i]:
            j = int(j)
            if 0 <= j < N:
                if exclude_self and j == i:
                    continue
                edges.add((i, j))

    if len(edges) == 0:
        return float("nan")

    mutual = 0
    for (i, j) in edges:
        if (j, i) in edges:
            mutual += 1

    return mutual / len(edges)


"""
load the binary mask and resize it to match the 224x224 res of the model 
"""
def load_mask_224(mask_path: str) -> np.ndarray:
    img = Image.open(mask_path).convert("L").resize((224, 224), resample=Image.NEAREST)
    m = np.array(img, dtype=np.uint8)
    m = (m > 127).astype(np.uint8) * 255
    return m


"""
Assumes the given grid is a square then figures out the grid height and width
"""
def infer_grid_from_N(N: int) -> tuple[int, int]:
    s = int(math.isqrt(N))
    if s * s != N:
        raise ValueError(f"N={N} is not a perfect square")
    return s, s


"""
load the KNN graph produced by the vig and return the neighbor index (shape is N, k)
"""
def load_knn_idx(pt_path: str) -> torch.Tensor:
    cap = torch.load(pt_path, map_location="cpu")
    if "knn_output" not in cap:
        raise KeyError(f"'knn_output' not found in {pt_path}. Keys: {list(cap.keys())}")

    knn = cap["knn_output"]

    if knn.ndim != 4 or knn.shape[0] != 2:
        raise ValueError(f"Expected knn_output shape (2, B, N, k). Got {tuple(knn.shape)}")
    if knn.shape[1] < 1:
        raise ValueError(f"Expected batch dim B>=1. Got {tuple(knn.shape)}")

    idx = knn[0, 0]
    return idx.long()


"""
Function to assign ice/ocean label to each node by majority pixel 
"""
def labels_majority_per_cell(mask224: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    H, W = mask224.shape
    if (H, W) != (224, 224):
        raise ValueError(f"Expected mask of shape (224,224), got {(H,W)}")

    cell_h = H // grid_h
    cell_w = W // grid_w
    if cell_h * grid_h != H or cell_w * grid_w != W:
        raise ValueError(f"Grid {grid_h}x{grid_w} must evenly divide 224x224.")

    labels = np.zeros(grid_h * grid_w, dtype=np.int64)

    for r in range(grid_h):
        for c in range(grid_w):
            idx = r * grid_w + c
            cell = mask224[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            white = np.count_nonzero(cell > 127)
            black = cell.size - white
            labels[idx] = 1 if white >= black else 0

    return labels


"""
This function computes the matching neighbors (ice vs ocean) for each node 
"""
def match_rates_from_idx(idx: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    N, k = idx.shape
    idx_np = idx.cpu().numpy()
    rates = np.zeros(N, dtype=np.float64)

    for i in range(N):
        nbrs = idx_np[i]
        nbrs = nbrs[(nbrs >= 0) & (nbrs < N)]
        nbrs = nbrs[nbrs != i]
        if nbrs.size == 0:
            rates[i] = np.nan
            continue
        rates[i] = float(np.mean(labels[nbrs] == labels[i]))

    return rates


"""
Load a graph, get the shape, do some metrics and print those metrics and save them for later use in plots
"""
def run_one(pt_path: str, mask224: np.ndarray, title: str, topk: int = 10, exclude_self: bool = True):
    print(f"\nGraph: {pt_path}")

    idx = load_knn_idx(pt_path)
    N = idx.shape[0]
    H, W = infer_grid_from_N(N)
    print(f"  Node grid: {H}x{W} (N={N})")

    stats = most_popular_neighbors(idx, topk=topk)
    counts = stats["counts"]
    top = stats["top"]
    total_slots = stats["total_slots"]

    best = int(top[0])
    best_count = int(counts[best])
    best_share = 100.0 * best_count / total_slots if total_slots > 0 else float("nan")

    print(f"  [{title}] Most popular neighbors:")
    print(f"    #1 neighbor = {best}  count={best_count}  share={best_share:.3f}% of non-self slots")
    print(f"    Top {topk}: " + ", ".join([f"{int(n)}({int(counts[int(n)])})" for n in top]))

    mrate = mutual_neighbor_rate(idx, exclude_self=exclude_self)
    print(f"  [{title}] Mutual neighbor rate: {100.0*mrate:.2f}%")

    node_labels = labels_majority_per_cell(mask224, H, W)
    rates = match_rates_from_idx(idx, node_labels)

    valid = rates[~np.isnan(rates)]
    mean_overall = float(np.mean(valid)) if valid.size else float("nan")

    ice = rates[(node_labels == 1) & ~np.isnan(rates)]
    ocean = rates[(node_labels == 0) & ~np.isnan(rates)]
    mean_ice = float(np.mean(ice)) if ice.size else float("nan")
    mean_ocean = float(np.mean(ocean)) if ocean.size else float("nan")

    print(f"  Overall: mean={100.0*mean_overall:.2f}%, (n={valid.size})")
    if ice.size:
        print(f"  ICE centers:   mean={100.0*mean_ice:.2f}%, (n={ice.size})")
    if ocean.size:
        print(f"  OCEAN centers: mean={100.0*mean_ocean:.2f}%, (n={ocean.size})")

    dmean = mean_neighbor_distances_from_idx(idx, H, W)
    d_valid = dmean[~np.isnan(dmean)]
    mean_dist_overall = float(np.mean(d_valid)) if d_valid.size else float("nan")

    d_ice = dmean[(node_labels == 1) & ~np.isnan(dmean)]
    d_ocean = dmean[(node_labels == 0) & ~np.isnan(dmean)]
    mean_dist_ice = float(np.mean(d_ice)) if d_ice.size else float("nan")
    mean_dist_ocean = float(np.mean(d_ocean)) if d_ocean.size else float("nan")

    print(f"  Mean neighbor distance in grid units: overall={mean_dist_overall:.3f}")
    if d_ice.size:
        print(f"    ICE centers:   mean={mean_dist_ice:.3f}")
    if d_ocean.size:
        print(f"    OCEAN centers: mean={mean_dist_ocean:.3f}")

    return {
        "title": title,
        "mean_overall_pct": 100.0 * mean_overall,
        "mean_ice_pct": 100.0 * mean_ice,
        "mean_ocean_pct": 100.0 * mean_ocean,
        "mutual_rate_pct": 100.0 * float(mrate),
        "mean_neighbor_dist_overall": mean_dist_overall,
        "mean_neighbor_dist_ice": mean_dist_ice,
        "mean_neighbor_dist_ocean": mean_dist_ocean,
        "grid_h": H,
        "grid_w": W,
    }


"""
Parses arguments and settings, loads binary mask, run and print metrics for specified graphs, create plots
"""
def main():
    ap = argparse.ArgumentParser(description="Compute neighborhood metrics from extracted ViG KNN graph files")

    ap.add_argument("--mask", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--graph-dir", nargs=2, action="append", metavar=("PATCH_NAME", "GRAPH_DIR"), required=True)
    ap.add_argument("--outdir", default="neighbor_metrics_plots")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--show_plots", action="store_true")
    ap.add_argument("--skip_grouped_plots", action="store_true")
    ap.add_argument("--make_individual_plots", action="store_true")

    args = ap.parse_args()

    mask224 = load_mask_224(args.mask)
    os.makedirs(args.outdir, exist_ok=True)

    blocks = args.blocks
    graph_dirs = {name: path for name, path in args.graph_dir}

    if not graph_dirs:
        raise ValueError("At least one --graph-dir PATCH_NAME GRAPH_DIR pair is required.")

    results_by_patch: dict[str, list[dict]] = {}

    for patch_name, graph_dir in graph_dirs.items():
        print(f"\n=== {patch_name} ===")
        patch_results = []

        for b in blocks:
            pt = os.path.join(graph_dir, f"block{b:02d}.pt")
            if not os.path.exists(pt):
                raise FileNotFoundError(pt)

            patch_results.append(
                run_one(
                    pt_path=pt,
                    mask224=mask224,
                    title=f"{patch_name} B{b:02d}",
                    topk=args.topk,
                )
            )

        add_normalized_distance(patch_results)
        results_by_patch[patch_name] = patch_results

    csv_path = os.path.join(args.outdir, "neighbor_metrics_summary.csv")

    rows = []

    for patch_name, results_list in results_by_patch.items():
        for d in results_list:
            block = block_num_from_title(d["title"])

            rows.append({
                "patch_name": patch_name,
                "block": block,
                "grid_h": d["grid_h"],
                "grid_w": d["grid_w"],
                "normalized_neighbor_distance_pct": d["mean_neighbor_dist_norm_pct"],
                "node_neighbor_label_correctness_pct": d["mean_overall_pct"],
                "node_neighbor_label_correctness_ice_pct": d["mean_ice_pct"],
                "node_neighbor_label_correctness_ocean_pct": d["mean_ocean_pct"],
                "mutual_neighbor_rate_pct": d["mutual_rate_pct"],
                "mean_neighbor_dist_overall": d["mean_neighbor_dist_overall"],
                "mean_neighbor_dist_ice": d["mean_neighbor_dist_ice"],
                "mean_neighbor_dist_ocean": d["mean_neighbor_dist_ocean"],
            })

    rows = sorted(rows, key=lambda x: (x["patch_name"], x["block"]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patch_name",
                "block",
                "grid_h",
                "grid_w",
                "normalized_neighbor_distance_pct",
                "node_neighbor_label_correctness_pct",
                "node_neighbor_label_correctness_ice_pct",
                "node_neighbor_label_correctness_ocean_pct",
                "mutual_neighbor_rate_pct",
                "mean_neighbor_dist_overall",
                "mean_neighbor_dist_ice",
                "mean_neighbor_dist_ocean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV summary to: {csv_path}")

    colors = {
        "P14": "#fbb4ae",
        "P56": "#b3cde3",
        "P112": "#ccebc5",
    }

    if not args.skip_grouped_plots:
        plot_grouped_bars_across_patches(
            results_by_patch=results_by_patch,
            blocks=blocks,
            metric_key="mean_overall_pct",
            ylabel="Label Correctness (%)",
            title="Node–Neighbor Label Correctness Across Blocks",
            outpath=os.path.join(args.outdir, "grouped_label_correctness_overall"),
            show=args.show_plots,
            colors=colors,
            value_labels=True,
            ylim=(0, 100),
        )

        plot_grouped_bars_across_patches(
            results_by_patch=results_by_patch,
            blocks=blocks,
            metric_key="mutual_rate_pct",
            ylabel="Mutual Neighbor Rate (%)",
            title="Mutual Neighbor Rate Across Blocks",
            outpath=os.path.join(args.outdir, "grouped_mutual_neighbor_rate"),
            show=args.show_plots,
            colors=colors,
            value_labels=True,
            ylim=(0, 100),
        )

        plot_grouped_bars_across_patches(
            results_by_patch=results_by_patch,
            blocks=blocks,
            metric_key="mean_neighbor_dist_norm_pct",
            ylabel="Normalized Neighbor Distance (%)",
            title="Normalized Neighbor Distance Across Blocks",
            outpath=os.path.join(args.outdir, "grouped_norm_neighbor_distance_overall"),
            show=args.show_plots,
            colors=colors,
            value_labels=True,
            ylim=(0, 100),
        )

    if args.make_individual_plots:
        for patch_name, results in results_by_patch.items():
            plot_means_across_blocks(
                results,
                patch_name=patch_name,
                outpath=os.path.join(args.outdir, f"{patch_name}_means_across_blocks.png"),
                show=args.show_plots
            )

            plot_mutual_rate_across_blocks(
                results,
                patch_name=patch_name,
                outpath=os.path.join(args.outdir, f"{patch_name}_mutual_rate_across_blocks.png"),
                show=args.show_plots
            )

            plot_neighbor_distance_across_blocks(
                results,
                patch_name=patch_name,
                outpath=os.path.join(args.outdir, f"{patch_name}_neighbor_distance_across_blocks.png"),
                show=args.show_plots,
            )


if __name__ == "__main__":
    main()