import argparse
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from neighbor_metrics import (
    most_popular_neighbors,
    load_mask_224,
    labels_majority_per_cell,
)
#---
#NOTE: This script creates graph visualizations from extracted ViG KNN graph files
#It is meant to be run through main.py using a YAML config with task: graph_visualization
#and run_mode: visualize.
#
#It supports two visualization options:
# 1. Most-popular-neighbor visualization:
#    Finds the node that appears most often as a neighbor across the graph.
#    The rank is controlled by --mp_top_rank.
#    For example, 1 = most popular neighbor, 2 = second most popular, 3 = third most popular.
#    It plots that hub node and all center nodes that include it in their neighbor list.
#
# 2. Center-neighbor visualization:
#    Uses the center node ids provided by --centers.
#    It plots each chosen center and the KNN neighbors connected to that center.
#
#The script expects:
#  --image: original SAR image
#  --mask: binary mask used to label nodes as ice/ocean for the MP visualization
#  --graph: extracted blockXX.pt file created by extract_graph.py
#  --block-name: label used for titles and output filenames
#  --outdir: folder where PNG/SVG figures are saved
#  --centers: node ids to visualize for the center-neighbor figure
#
#Use --skip_mp to skip the MP visualization.
#Use --skip_centers to skip the chosen center-neighbor visualization.
#Use --show_plots to display figures interactively; otherwise figures are only saved.
#---


CENTER_COLORS = ["blue", "green", "orange", "purple", "cyan", "magenta"]

RESIZE_TO = (224, 224)      #224 is the size the model uses
SHOW_GRID = True            #make sure I can see the grid of patches
SHOW_NODE_LABELS = False    #don't need to see node labels right now
DRAW_EDGES = True           #want to see the neighbor connections for chosen centers
COLOR = "blue"

#matplot settings for the markers
CENTER_MARKER = "s"
CENTER_SIZE = 40
NEIGHBOR_SIZE = 40

#settings for the most popular neighbor plots
MOST_POP = "s"
CENTER_MARKS = 'o'
ICE_COLOR = "blue"
OCEAN_COLOR ="green"


def norm_title(s: str) -> str:
    return s.strip()


def idx_to_rowcol(idx: int, W: int):
    return idx // W, idx % W


#Function to plot the most popular nieghbor and it's related centers
def plot_popular_neighbor_and_center(ax, img_np, knn_idx, node_labels, title, top_rank=3):
    #Get grid and cell size
    N, k = knn_idx.shape
    H, W = infer_grid_from_N(N)
    Hpx, Wpx = img_np.shape[0], img_np.shape[1]
    cell = Wpx / W

    #grab most popular neigbhroe info
    stats = most_popular_neighbors(knn_idx, topk=max(top_rank, 1))
    counts = stats["counts"]
    total_slots = stats["total_slots"]

    #set up most popular stats
    mp = int(stats["top"][top_rank - 1])
    mp_count = int(counts[mp]) if counts is not None else 0
    mp_share = (100.0 * mp_count / total_slots) if total_slots > 0 else float("nan")

    #get all the centers that have that most popular neighbor
    idx_np = knn_idx.cpu().numpy()
    centers = np.nonzero((idx_np == mp).any(axis=1))[0]
    centers = centers[centers != mp]

    #need to see the centers to plot them on the other vis
    centers_list = centers.astype(int).tolist()
    print(f"[{title}] Most-pop neighbor (hub) = {mp}")
    print(f"[{title}] Centers that include hub in their neighbor list (count={len(centers_list)}):")
    print(centers_list)

    #draw image and grid, fix bounds so grid overlay looks good
    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis("off")
    ax.set_xlim(0, Wpx)
    ax.set_ylim(Hpx, 0)
    ax.set_aspect("equal")
    if SHOW_GRID:
        for i in range(H + 1):
            ax.axhline(i * cell, linewidth=0.3, clip_on=True)
        for j in range(W + 1):
            ax.axvline(j * cell, linewidth=0.3, clip_on=True)

    #scale the size based on the cell so it looks right on both patch sizes
    mp_size = max(20.0, (cell * 1.2) ** 2)
    center_size = max(10.0, (cell * 0.8) ** 2)

    #color the MP depending on if it's ice or ocean (ice blue, ocean green)
    #mp_is_ice = bool(node_labels[mp] == 1)
    #mp_color = ICE_COLOR if mp_is_ice else OCEAN_COLOR
    #red for now so I can see it
    mp_color = 'red'

    #Plot the most popular neighbor
    hr, hc = node_to_rc(mp, W)
    hx, hy = rc_to_xy(hr, hc, cell)
    ax.scatter(
        [hx], [hy],
        s=40,
        marker=MOST_POP,
        color=mp_color,
        zorder=7,
        label="Most popular neighbor"
    )

    #split the centers that point to MP into ocean and ice
    if centers.size:
        centers_ice = centers[node_labels[centers] == 1]
        centers_ocean = centers[node_labels[centers] == 0]

        #helper to plot the ice and ocean nodes with the proper color, marker, and location
        def scatter_centers(nodes, color, label):
            if nodes.size == 0:
                return
            xs, ys = [], []
            for n in nodes:
                r, c = node_to_rc(int(n), W)
                x, y = rc_to_xy(r, c, cell)
                xs.append(x);
                ys.append(y)
                #ax.text( x + cell * 0.15, y - cell * 0.15, str(int(n)), fontsize=7,color="black",
                 #   ha="left", va="top", zorder=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),)
            ax.scatter(xs, ys, s=40, marker=CENTER_MARKS, color=color,
                       alpha=0.80, zorder=6, label=label)

        scatter_centers(centers_ice, ICE_COLOR, "Ice Centers of MP")
        scatter_centers(centers_ocean, OCEAN_COLOR, "Ocean Centers of MP")

    #add legend
    legend_handles = [
        Line2D([0], [0], marker=MOST_POP, linestyle="None",
               markerfacecolor=mp_color, markeredgecolor="black",
               markersize=10, label="MP neighbor"),
        Line2D([0], [0], marker=CENTER_MARKS, linestyle="None",
               markerfacecolor=ICE_COLOR, markeredgecolor="black",
               markersize=9, label="Ice centers of MP"),
        Line2D([0], [0], marker=CENTER_MARKS, linestyle="None",
               markerfacecolor=OCEAN_COLOR, markeredgecolor="black",
               markersize=9, label="Ocean centers of MP"),
    ]

    # Place legend outside, below the plot
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),  # below axes
        ncol=3,
        frameon=False,
        fontsize=14,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # Stats line outside, below the legend (paper-friendly, compact)
    # fraction as count/total plus percent in parentheses
    stats_text = (
        f"Share of Neighbor Slots={mp_count}/{total_slots} ({mp_share:.2f}%)  |  Connected Centers={centers.size}"
    )

    ax.text(
        0.5, -0.14,  # below legend
        stats_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=14,
        clip_on=False
    )

    print(f"[{title}] hub={mp}, centers={centers.size}, hub_share={mp_share:.2f}%")


#open the extracted graph, check that it has knn tensor and get the tensor if it does
def load_capture(pt_path: str) -> torch.Tensor:
    cap = torch.load(pt_path, map_location="cpu")
    if "knn_output" not in cap:
        raise KeyError(f"'knn_output' not found in {pt_path}. Keys: {list(cap.keys())}")
    knn = cap["knn_output"]
    if not torch.is_tensor(knn):
        raise TypeError(f"knn_output is not a tensor. Type: {type(knn)}")
    return knn


#get the H/W of the grid based on N, only works if it's a square (it should be)
def infer_grid_from_N(N: int) -> tuple[int, int]:
    s = int(math.isqrt(N))
    if s * s != N:
        raise ValueError(
            f"N={N} is not a perfect square"
        )
    return s, s


#remap the manually chosen centers to work on different patch sizes
def remap_centers(old_centers, old_W, new_W):
    new_centers = []
    for n in old_centers:
        r, c = (n // old_W, n % old_W)

        rr = r / (old_W - 1)
        cc = c / (old_W - 1)

        r2 = int(round(rr * (new_W - 1)))
        c2 = int(round(cc * (new_W - 1)))

        new_centers.append(r2 * new_W + c2)
    return new_centers


#based on the node, get the row and column for the grid in the vis
def node_to_rc(n: int, W: int) -> tuple[int, int]:
    return (n // W, n % W)


#based on grid cell, get the pixel coords at the center of the cell
def rc_to_xy(r: int, c: int, cell: float) -> tuple[float, float]:
    return ((c + 0.5) * cell, (r + 0.5) * cell)


#based on graph and chosen center, get a list of the KNNs
def neighbors_for_center(knn: torch.Tensor, center: int) -> list[int]:
    #knn shape (2(s/t), B, N, k).
    #knn[0, 0, center, :] are the neighbor indices for that center
    if knn.dim() != 4 or knn.shape[0] != 2:
        raise ValueError(f"Expected knn shape (2, B, N, k). Got: {tuple(knn.shape)}")
    if knn.shape[1] < 1:
        raise ValueError(f"Expected batch dim B>=1. Got: {tuple(knn.shape)}")

    N = knn.shape[2]
    if not (0 <= center < N):
        raise ValueError(f"Center {center} out of range 0..{N-1}")

    return knn[0, 0, center].tolist()


#plot the chosen centers and it's neighbors
def plot_center_and_neighbors(ax, img_np: np.ndarray, knn: torch.Tensor, title: str, centers_for_block: list[int]):
    #get the grid dimensions
    _, B, N, k = knn.shape
    H, W = infer_grid_from_N(N)

    legend_x = 0.02
    legend_y = 0.98
    line_spacing = 0.05

    #get cell size in pixels, 224/28 = 8
    Hpx, Wpx = img_np.shape[0], img_np.shape[1]
    cell = Wpx / W

    ax.imshow(img_np)
    ax.set_title(title)
    ax.axis("off")
    #set the h/w and make the grid match the image bounds so there's not overhang
    Hpx, Wpx = img_np.shape[0], img_np.shape[1]
    ax.set_xlim(0, Wpx)
    ax.set_ylim(Hpx, 0)
    ax.set_aspect("equal")

    #draw the grid overlay
    if SHOW_GRID:
        for i in range(H + 1):
            ax.axhline(i * cell, linewidth=0.3, clip_on=True)
        for j in range(W + 1):
            ax.axvline(j * cell, linewidth=0.3, clip_on=True)

    for i, center in enumerate(centers_for_block, start=1):
        color = CENTER_COLORS[(i - 1) % len(CENTER_COLORS)]
        center = int(center)
        #get neighbors
        neighbors = neighbors_for_center(knn, center)
        #removing the self neighbor so we can see the center w/o overlap, the paper is the same
        neighbors = [nb for nb in neighbors if nb != center]

        #get the pixl coords of the center node
        cr, cc = node_to_rc(center, W)
        cx, cy = rc_to_xy(cr, cc, cell)
        print(f"[{title}] center={center} -> (row={cr}, col={cc}) | pixel≈({cx:.1f},{cy:.1f})")

        #get the pixel coords of the neigbors and draw edges to the center node
        nxy = []
        for nb in neighbors:
            nr, nc = node_to_rc(nb, W)
            nx, ny = rc_to_xy(nr, nc, cell)
            if DRAW_EDGES:
                ax.plot([cx, nx], [cy, ny], linewidth=1, zorder=5, color=color)
            nxy.append((nx, ny, nb))

        # draw points on top
        if nxy:
            ax.scatter([p[0] for p in nxy], [p[1] for p in nxy], s=NEIGHBOR_SIZE, color= color, zorder=6)

        ax.scatter([cx], [cy], s=CENTER_SIZE, marker=CENTER_MARKER, color=color, zorder=7)

        #currently off, but if I want to see the node labels next to the neigbors
        if SHOW_NODE_LABELS:
            ax.text(cx + 2, cy + 2, str(center), fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        #details to see in the console of the center and it's neighbors
        print(f"[{title}] center={center} neighbors (k={k}): {neighbors}")

    center_handles = []
    for i, center in enumerate(centers_for_block, start=1):
        color = CENTER_COLORS[(i - 1) % len(CENTER_COLORS)]
        center_handles.append(
            Line2D([0], [0],
                   marker=CENTER_MARKER, linestyle="None",
                   markerfacecolor=color, markeredgecolor="black",
                   markersize=10,
                   label=f"Center {int(center)}")
        )

    ax.legend(
        handles=center_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(center_handles),
        frameon=False,
        fontsize=14,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # label for each vis to debug if needed
    #ax.text(
     #   5, 15, f"N={N} ({H}x{W}), k={k}, centers={MANUAL_CENTERS}",
      #  fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    #)


def main():
    ap = argparse.ArgumentParser(description="Create graph visualizations from extracted ViG KNN graph files")

    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--block-name", default="Block")
    ap.add_argument("--outdir", default="graph_vis_outputs")
    ap.add_argument("--centers", type=int, nargs="+", default=[])
    ap.add_argument("--show_plots", action="store_true")

    #keeps the older figure options available without hardcoding them
    ap.add_argument("--skip_mp", action="store_true")
    ap.add_argument("--skip_centers", action="store_true")
    ap.add_argument("--mp_top_rank", type=int, default=3)

    args = ap.parse_args()

    if not args.skip_centers and not args.centers:
        raise ValueError("--centers is required unless --skip_centers is used.")

    if args.skip_mp and args.skip_centers:
        raise ValueError("Both visualizations are disabled. Enable at least one.")

    block_name = args.block_name
    pt_path = args.graph
    outdir = args.outdir
    centers_for_block = args.centers

    os.makedirs(outdir, exist_ok=True)

    # check inputs exist
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    if not os.path.exists(args.mask):
        raise FileNotFoundError(args.mask)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(pt_path)

    # load image and resize
    img = Image.open(args.image).convert("RGB").resize(RESIZE_TO)
    img_np = np.array(img)

    # load mask
    mask224 = load_mask_224(args.mask)

    # load graph capture (knn tensor shape (2, B, N, k))
    knn = load_capture(pt_path)
    knn_idx = knn[0, 0]  # (N, k)

    # compute labels for this graph's grid resolution
    N = knn_idx.shape[0]
    H, W = infer_grid_from_N(N)
    node_labels = labels_majority_per_cell(mask224, H, W)

    safe_block_name = block_name.lower().replace(" ", "_")

    # Figure 1: MP visualization
    if not args.skip_mp:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

        plot_popular_neighbor_and_center(
            ax=ax1,
            img_np=img_np,
            knn_idx=knn_idx,
            node_labels=node_labels,
            title=block_name,
            top_rank=args.mp_top_rank,
        )

        fig1.tight_layout()
        fig1.savefig(os.path.join(outdir, f"{safe_block_name}_mp.png"), dpi=300, bbox_inches="tight")
        fig1.savefig(os.path.join(outdir, f"{safe_block_name}_mp.svg"), bbox_inches="tight")

        # Optional: show MP figure
        if args.show_plots:
            plt.show()
        plt.close(fig1)

    # Figure 2: centers + neighbors visual
    if not args.skip_centers:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

        plot_center_and_neighbors(
            ax=ax2,
            img_np=img_np,
            knn=knn,
            title=block_name,
            centers_for_block=centers_for_block,
        )

        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir, f"{safe_block_name}_centers.png"), dpi=300, bbox_inches="tight")
        fig2.savefig(os.path.join(outdir, f"{safe_block_name}_centers.svg"), bbox_inches="tight")

        # Optional: show centers figure
        if args.show_plots:
            plt.show()
        plt.close(fig2)


if __name__ == "__main__":
    main()