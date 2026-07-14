import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from timm.models import create_model

import vig
from vig import load_pretrained_pos_embed
from gcn_lib import Grapher

#get a list of grapher modules so i can pick which block later, should be 16 for vigs
def find_graphers(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, Grapher)]


#load the image and match the default cfgs from vig.py
#model expects RGB size 224
def load_image_tensor(img_path: str, img_size: int = 224, in_chans: int = 3):
    if in_chans == 1:
        img = Image.open(img_path).convert("L")
        mean = (0.5,)
        std = (0.5,)
    elif in_chans == 3:
        img = Image.open(img_path).convert("RGB")
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"in_chans must be 1 or 3, got {in_chans}")

    tfm = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return tfm(img).unsqueeze(0)

#build the model, supports multiple models, but I'm using ViGs
#i'm using imagenet weights so it expects 1000 for num classes
def build_model(
    model_name: str,
    img_size: int = 224,
    grid_size: int = 56,
    in_chans: int = 3,
):
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        img_size=img_size,
        grid_size=grid_size,
        in_chans=in_chans,
        classifier_mode="image",
    )
    return model

#load the weights, I'm using imagenet currently for vigS, and make sure they actually worked
def load_weights_into(model, weights_path: str | None):
    if not weights_path:
        print("No pretrain_path provided. Using randomly initialized weights.")
        return

    print(f"Loading weights from: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    load_pretrained_pos_embed(model, state)

    print("Weights loaded.")


@torch.no_grad()
def extract_knn_graph_once(
    model,
    x,
    which_grapher: int,
    device: str,
    save_path: str | None,
):
    model = model.to(device).eval()
    x = x.to(device)

    # find graphers and make sure the chosen grapher is in the right range
    graphers = find_graphers(model)
    if not graphers:
        raise RuntimeError("No Grapher modules found.")
    if which_grapher < 0 or which_grapher >= len(graphers):
        raise ValueError(
            f"which_grapher out of range. Must be 0..{len(graphers) - 1}"
        )

    # grapher selection and knn module to be used
    # this is where the edge index is in grapher: edge_index = self.dilated_knn_graph(x, y, relative_pos)
    name, grapher = graphers[which_grapher]
    knn_mod = grapher.graph_conv.dilated_knn_graph

    # saving what was captured to print out later
    captured = {
        "grapher_name": name,
        "which_grapher": which_grapher,
        "block_number": which_grapher + 1,
        "hooked_module": "graph_conv.dilated_knn_graph",
        "edge_index": None,
        "knn_output": None,
        "knn_input_shapes": [],
    }

    # forward hook for densedilatedknngraph to get the shape/type of inputs used in the knn and get the edge index
    def hook(mod, inputs, output):
        for inp in inputs:
            if torch.is_tensor(inp):
                captured["knn_input_shapes"].append(tuple(inp.shape))
            else:
                captured["knn_input_shapes"].append(type(inp).__name__)

        captured["edge_index"] = output
        captured["knn_output"] = output

    # do one capture then remove the hook
    h = knn_mod.register_forward_hook(hook)
    _ = model(x)
    h.remove()

    # make sure I captured something, detach it, move to cpu
    edge_index = captured["edge_index"]
    if edge_index is None:
        raise RuntimeError("No KNN output captured.")

    captured["edge_index"] = edge_index.detach().cpu()
    captured["knn_output"] = edge_index.detach().cpu()

    # save the pt file for vis later
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captured, save_path)
        captured["saved_path"] = str(save_path)

    return captured

#print out a summary of the knn captured
def summarize_knn_capture(captured):
    print("\n==================== KNN CAPTURE SUMMARY ====================")
    print("Block:", captured["block_number"])
    print("Grapher:", captured["grapher_name"])
    print("Hooked:", captured["hooked_module"])
    print("KNN input shapes/types:", captured["knn_input_shapes"])

    edge_index = captured["edge_index"]

    if torch.is_tensor(edge_index):
        print("edge_index shape:", tuple(edge_index.shape), "dtype:", edge_index.dtype)

        if edge_index.dim() == 4 and edge_index.shape[0] == 2:
            _, B, N, k = edge_index.shape
            print(f"Interpreted: B={B}, N={N} nodes, k={k} neighbors per node")
    else:
        print("edge_index type:", type(edge_index).__name__)

    if "saved_path" in captured:
        print("Saved to:", captured["saved_path"])

    print("============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Extract ViG KNN graphs from selected Grapher blocks")

    parser.add_argument("--image", required=True, help="Path to input SAR image")
    parser.add_argument("--model", default="vig_s_224_gelu", help="Model name registered with timm")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=56, choices=[7, 14, 28, 56, 112])
    parser.add_argument("--in-chans", type=int, default=3, choices=[1, 3])
    parser.add_argument("--pretrain_path", type=str, default=None)

    parser.add_argument("--blocks", type=int, nargs="+", default=[1])
    parser.add_argument("--save-dir", type=str, default="outputs/graphs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; switching to CPU.")
        device = "cpu"

    # load image/weights, build model, find graphers
    x = load_image_tensor(
        img_path=args.image,
        img_size=args.img_size,
        in_chans=args.in_chans,
    )

    model = build_model(
        model_name=args.model,
        img_size=args.img_size,
        grid_size=args.grid_size,
        in_chans=args.in_chans,
    )

    load_weights_into(model, args.pretrain_path)

    graphers = find_graphers(model)
    print(f"Found {len(graphers)} Grapher blocks.")

    for block in args.blocks:
        which_grapher = block - 1

        save_path = Path(args.save_dir) / f"block{block:02d}.pt"

        # run the model on the image, hook to the specified grapher, save what was captured
        captured = extract_knn_graph_once(
            model=model,
            x=x,
            which_grapher=which_grapher,
            device=device,
            save_path=str(save_path),
        )

        summarize_knn_capture(captured)


if __name__ == "__main__":
    main()