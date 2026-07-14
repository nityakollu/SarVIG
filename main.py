"""
This will be the entry point for the updated pipeline allowing me to choose mode and set params use configs
"""
import argparse
import subprocess
import sys
import yaml

"""
Load the config file and turn it into a dictionary 
"""
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

"""
Build the command for training/finetuning/eval for image or patch classification tasks
"""
def build_train_command(cfg: dict, config_path: str) -> list:
    cmd = [sys.executable, "train.py", "--config", config_path]

    data_dir = cfg["data"]["root"]
    cmd.append(data_dir)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})
    task = cfg["task"]
    run_mode = cfg["run_mode"]

    #Currently supporting image and patch level classification and self supervised training
    if task == "image_classification":
        classifier_mode = "image"
    elif task == "patch_classification":
        classifier_mode = "patch"
    elif task == "self_supervised_train":
        classifier_mode = "self_supervised"
    else:
        raise ValueError(f"Task '{task}' unknown or not supported ")

    #Required args, defaults set if not given
    cmd += ["--model", model_cfg.get("name", "vig_s_224_gelu")]
    cmd += ["--classifier-mode", classifier_mode]
    cmd += ["--in-chans", str(model_cfg.get("in_chans", 3))]

    # set to 224 in config, DO NOT CHANGE! the og model is built around that size
    if model_cfg.get("img_size") is not None:
        cmd += ["--img-size", str(model_cfg["img_size"])]
    if model_cfg.get("grid_size") is not None:
        cmd += ["--grid-size", str(model_cfg.get("grid_size", 56))]

    #only classification uses num classes
    if task in {"image_classification", "patch_classification"}:
        cmd += ["--num-classes", str(model_cfg.get("num_classes", 2))]

    #set the given pretrain path
    if model_cfg.get("pretrain_path"):
        cmd += ["--pretrain_path", model_cfg["pretrain_path"]]

    if model_cfg.get("resume"):
        cmd += ["--resume", model_cfg["resume"]]

    # Freezing args
    if model_cfg.get("freeze_mode") is not None:
        cmd += ["--freeze-mode", str(model_cfg.get("freeze_mode", "none"))]

    if model_cfg.get("freeze_blocks") is not None:
        cmd += ["--freeze-blocks", str(model_cfg.get("freeze_blocks", 0))]

    #Training args, defaults currently match the og model which is based on ImageNet
    cmd += ["--batch-size", str(train_cfg.get("batch_size", 1))]
    cmd += ["--epochs", str(train_cfg.get("epochs", 1))]
    cmd += ["--lr", str(train_cfg.get("lr", 0.01))]
    cmd += ["--opt", train_cfg.get("optimizer", "sgd")]
    cmd += ["--weight-decay", str(train_cfg.get("weight_decay", 0.0001))]
    cmd += ["--sched", train_cfg.get("scheduler", "step")]

    if train_cfg.get("warmup-epochs") is not None:
        cmd += ["--warmup-epochs", str(train_cfg.get("warmup-epochs"))]
    if train_cfg.get("warmup-lr") is not None:
        cmd += ["--warmup-lr", str(train_cfg.get("warmup-lr"))]

    cmd += ["--workers", str(runtime_cfg.get("workers", 0))]
    cmd += ["--output", runtime_cfg.get("output_dir", "./output")]
    cmd += ["--seed", str(runtime_cfg.get("seed", 42))]

    #Ensures aug turned off for patch classification
    if task == "patch_classification":
        cmd += ["--smoothing", "0.0"]
        cmd += ["--mixup", "0.0"]
        cmd += ["--cutmix", "0.0"]
        cmd.append("--no-prefetcher")

    elif task == "self_supervised":
        cmd.append("--no-prefetcher")

    #If in training only, skip validation
    if run_mode == "train":
        cmd.append("--train-only")
        cmd.append("--skip-val")
    #Both training and validation occur here for finetuning
    elif run_mode == "train_eval":
        pass
    #this is the final eval mode for patches
    elif run_mode == "eval":
        cmd.append("--evaluate")
    else:
        raise ValueError(f"Unsupported run_mode for train.py: {run_mode}")

    return cmd

"""
Build the command for KNN graph extraction
"""
def build_graph_extract_command(cfg: dict) -> list:
    cmd = [sys.executable, "extract_graph.py"]

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    extract_cfg = cfg.get("extraction", {})
    runtime_cfg = cfg.get("runtime", {})

    cmd += ["--image", data_cfg["image"]]
    cmd += ["--model", model_cfg.get("name", "vig_s_224_gelu")]
    cmd += ["--grid-size", str(model_cfg.get("grid_size", 56))]
    cmd += ["--in-chans", str(model_cfg.get("in_chans", 3))]

    if model_cfg.get("img_size") is not None:
        cmd += ["--img-size", str(model_cfg["img_size"])]

    if model_cfg.get("pretrain_path"):
        cmd += ["--pretrain_path", model_cfg["pretrain_path"]]

    if extract_cfg.get("blocks"):
        cmd += ["--blocks", *[str(b) for b in extract_cfg["blocks"]]]

    if extract_cfg.get("save_dir"):
        cmd += ["--save-dir", extract_cfg["save_dir"]]

    cmd += ["--device", runtime_cfg.get("device", "cuda")]
    cmd += ["--seed", str(runtime_cfg.get("seed", 42))]

    return cmd

"""
Build the command for neighborhood metrics
"""
def build_neighbor_metrics_command(cfg: dict) -> list:
    cmd = [sys.executable, "neighbor_metrics.py"]

    data_cfg = cfg.get("data", {})
    metrics_cfg = cfg.get("metrics", {})

    cmd += ["--mask", data_cfg["mask"]]

    blocks = metrics_cfg.get("blocks", [1, 2, 3, 4, 5, 6, 12])
    graph_dirs = metrics_cfg.get("graph_dirs", {})

    for patch_name, graph_dir in graph_dirs.items():
        patch_lower = patch_name.lower()

        for b in blocks:
            block_arg = f"--{patch_lower}_block{b:02d}"
            block_path = f"{graph_dir}/block{b:02d}.pt"
            cmd += [block_arg, block_path]

    cmd += ["--outdir", metrics_cfg.get("outdir", "neighbor_metrics_plots")]

    if metrics_cfg.get("show_plots", False):
        cmd.append("--show_plots")

    return cmd


"""
Build the command for graph visualization
"""
def build_graph_vis_command(cfg: dict) -> list:
    cmd = [sys.executable, "graph_vis.py"]

    data_cfg = cfg.get("data", {})
    vis_cfg = cfg.get("visualization", {})

    cmd += ["--image", data_cfg["image"]]
    cmd += ["--mask", data_cfg["mask"]]
    cmd += ["--graph", vis_cfg["graph_path"]]
    cmd += ["--block-name", vis_cfg.get("block_name", "Block")]
    cmd += ["--outdir", vis_cfg.get("outdir", "graph_vis_outputs")]
    cmd += ["--mp_top_rank", str(vis_cfg.get("mp_top_rank", 3))]

    if vis_cfg.get("centers"):
        cmd += ["--centers", *[str(c) for c in vis_cfg["centers"]]]

    if vis_cfg.get("show_plots", False):
        cmd.append("--show_plots")

    if not vis_cfg.get("make_mp", True):
        cmd.append("--skip_mp")

    if not vis_cfg.get("make_centers", True):
        cmd.append("--skip_centers")

    return cmd

def main():
    parser = argparse.ArgumentParser(description="SARViG pipeline entry point")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    #add dry run to run configs to print the command and NOT run the command
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    args = parser.parse_args()

    cfg = load_config(args.config)

    task = cfg["task"]
    run_mode = cfg["run_mode"]

    #classification tasks
    if task in {"image_classification", "patch_classification", "self_supervised_train"}:
        cmd = build_train_command(cfg, args.config)

    #graph extraction task
    elif task == "graph_extraction":
        if run_mode != "extract":
            raise ValueError("graph_extraction should use run_mode: extract")
        cmd = build_graph_extract_command(cfg)

    #neighborhood metrics task
    elif task == "neighbor_metrics":
        if run_mode != "metrics":
            raise ValueError("neighbor_metrics should use run_mode: metrics")
        cmd = build_neighbor_metrics_command(cfg)

    #graph visualization task
    elif task == "graph_visualization":
        if run_mode != "visualize":
            raise ValueError("graph_visualization should use run_mode: visualize")
        cmd = build_graph_vis_command(cfg)

    else:
        raise ValueError(f"Unknown task: {task}")

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()