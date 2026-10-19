import os
import sys
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer, set_seed

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose

# ----------------------------------------------------
# Integrated 3D-PointCloud-Aligner with Auto-Recovery for Colab/Kaggle
# ----------------------------------------------------
ALIGNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D-PointCloud-Aligner")
ALIGNER_SRC = os.path.join(ALIGNER_DIR, "src")


def _ensure_aligner_available():
    """Automatically initializes or clones the submodule if missing (e.g. fresh git clone or Colab git pull)."""
    target_file = os.path.join(ALIGNER_DIR, "src", "pipeline.py")
    if os.path.exists(target_file):
        return

    print("[Pipeline Notice] 3D-PointCloud-Aligner submodule not initialized. Auto-fetching...")
    import subprocess
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # 1. Try git submodule update --init --recursive
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except Exception:
        pass

    # 2. If still missing, clone repository directly into ALIGNER_DIR
    if not os.path.exists(target_file):
        try:
            os.makedirs(ALIGNER_DIR, exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://github.com/A511-git/3D-PointCloud-Aligner.git", ALIGNER_DIR],
                check=True,
                capture_output=True,
            )
            print("[Pipeline Notice] Successfully fetched 3D-PointCloud-Aligner!")
        except Exception as e:
            print(f"[Pipeline Warning] Could not auto-fetch 3D-PointCloud-Aligner: {e}")


_ensure_aligner_available()

for p in [ALIGNER_DIR, ALIGNER_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.pipeline import PointCloudPipeline
    from src.validator import PointCloudValidator
except ModuleNotFoundError:
    try:
        from pipeline import PointCloudPipeline
        from validator import PointCloudValidator
    except ModuleNotFoundError:
        PointCloudPipeline = None
        PointCloudValidator = None

from render_floorplan import render_floorplan

DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    code_template_file,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    seed=-1,
    max_new_tokens=4096,
    detect_type="all",
    categories=[],
):
    if seed >= 0:
        set_seed(seed)

    # load the code template
    with open(code_template_file, "r") as f:
        code_template = f.read()

    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    print("Task prompt: ", task_prompt)

    prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt} The reference code is as followed: {code_template}"

    # prepare the conversation data
    if model.config.model_type == "spatiallm_qwen":
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    do_sample = (temperature > 0.05) and (num_beams == 1)
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        point_clouds=point_cloud,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        num_beams=num_beams,
    )

    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
        generate_kwargs["top_k"] = int(top_k)

    if num_beams == 1:
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs["streamer"] = streamer
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        print("Generating layout...\n")
        generate_texts = []
        for text in streamer:
            generate_texts.append(text)
            print(text, end="", flush=True)
        print("\nDone!")
        t.join()
        layout_str = "".join(generate_texts)
    else:
        print(f"Generating layout with beam search (num_beams={num_beams})...\n")
        outputs = model.generate(**generate_kwargs)
        new_tokens = outputs[0][input_ids.shape[1] :]
        layout_str = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(layout_str)
        print("\nDone!")

    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpatialLM inference script")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file or a folder containing multiple point cloud files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output layout txt file or a folder to save multiple layout txt files",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM-Llama-1B",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        type=str,
        default="all",
        choices=["all", "arch", "object"],
        help="The type of indoor elements to detect. all: (wall, door, window, box), arch: (wall, door, window), object: (box)",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        choices=[
            "sofa",
            "chair",
            "dining_chair",
            "bar_chair",
            "stool",
            "bed",
            "pillow",
            "wardrobe",
            "nightstand",
            "tv_cabinet",
            "wine_cabinet",
            "bathroom_cabinet",
            "shoe_cabinet",
            "entrance_cabinet",
            "decorative_cabinet",
            "washing_cabinet",
            "wall_cabinet",
            "sideboard",
            "cupboard",
            "coffee_table",
            "dining_table",
            "side_table",
            "dressing_table",
            "desk",
            "integrated_stove",
            "gas_stove",
            "range_hood",
            "micro-wave_oven",
            "sink",
            "stove",
            "refrigerator",
            "hand_sink",
            "shower",
            "shower_room",
            "toilet",
            "tub",
            "illumination",
            "chandelier",
            "floor-standing_lamp",
            "wall_decoration",
            "painting",
            "curtain",
            "carpet",
            "plants",
            "potted_bonsai",
            "tv",
            "computer",
            "air_conditioner",
            "washing_machine",
            "clothes_rack",
            "mirror",
            "bookcase",
            "cushion",
            "bar",
            "screen",
            "combination_sofa",
            "dining_table_combination",
            "leisure_table_and_chair_combination",
            "multifunctional_combination_bed",
        ],
        help="A list of categories of objects to detect. If not specified, all categories will be detected.",
    )
    DEFAULT_CODE_TEMPLATE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "code_template.txt"
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default=DEFAULT_CODE_TEMPLATE,
        help="Path to the code template file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for inference ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="The number of highest probability vocabulary tokens to keep for top-k filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="bfloat16",
        help="The torch dtype to use for inference, bfloat16 or float32",
    )
    parser.add_argument(
        "--no_cleanup",
        default=False,
        action="store_true",
        help="Whether to not cleanup the point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The seed to use during inference, negative value means no seed",
    )

    # ----------------------------------------------------
    # Integrated 3D Point Cloud Alignment & Validation CLI
    # ----------------------------------------------------
    align_group = parser.add_argument_group("Integrated 3D PointCloud Aligner Options")
    align_group.add_argument(
        "--align",
        action="store_true",
        help="Automatically align input point cloud to ScanNet convention (+Z up, Manhattan walls, metric scale)",
    )
    align_group.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["arkit", "opencv", "opengl", "scannet"],
        help="Camera sensor preset convention (e.g. arkit/opengl = +Y up, opencv = -Y up)",
    )
    align_group.add_argument(
        "--up_axis",
        type=str,
        default="auto",
        choices=["auto", "x", "-x", "y", "-y", "z", "-z"],
        help="Vertical axis direction for alignment (default: auto)",
    )
    align_group.add_argument(
        "--target_height",
        type=float,
        default=2.65,
        help="Target floor-to-ceiling clearance in meters for metric scale calibration",
    )
    align_group.add_argument(
        "--force_scale",
        action="store_true",
        help="Force scaling to target_height even if detected height is within metric range",
    )
    align_group.add_argument(
        "--preview",
        action="store_true",
        help="Generate 3-view orthogonal diagnostic PNG preview of the aligned point cloud",
    )
    align_group.add_argument(
        "--save_aligned_pcd",
        type=str,
        default=None,
        help="Optional path to save the aligned point cloud (.ply) to disk",
    )
    align_group.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip pre-flight point cloud quality & orientation validation check",
    )

    # ----------------------------------------------------
    # Integrated Architectural Blueprint Floor Plan Rendering
    # ----------------------------------------------------
    render_group = parser.add_argument_group("Integrated Floor Plan Blueprint Options")
    render_group.add_argument(
        "--render_floorplan",
        type=str,
        default=None,
        help="Path to save the rendered 2D architectural blueprint floor plan PNG (e.g. floorplan.png)",
    )
    args = parser.parse_args()

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.inference_dtype)
    )
    model.to(args.device)
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # number of bins used for discretization
    num_bins = model.config.point_config["num_bins"]

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))

    for point_cloud_file in tqdm(point_cloud_files):
        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        active_pcd_path = point_cloud_file

        # ----------------------------------------------------
        # 1. Integrated 3D Point Cloud Alignment & Validation
        # ----------------------------------------------------
        should_align = args.align or (args.preset is not None)
        if should_align:
            print(f"\n[Pipeline] Aligning '{os.path.basename(point_cloud_file)}' to ScanNet convention...")
            preview_file = None
            if args.preview:
                if args.save_aligned_pcd:
                    preview_file = os.path.splitext(args.save_aligned_pcd)[0] + "_preview.png"
                elif os.path.splitext(args.output)[-1]:
                    preview_file = os.path.splitext(args.output)[0] + "_preview.png"
                else:
                    preview_file = os.path.splitext(point_cloud_file)[0] + "_aligned_preview.png"

            if PointCloudPipeline is None:
                raise ImportError(
                    "3D-PointCloud-Aligner could not be loaded. "
                    "Run 'git submodule update --init --recursive' or 'pip install git+https://github.com/A511-git/3D-PointCloud-Aligner.git'."
                )

            align_pipeline = PointCloudPipeline(
                up_axis=args.up_axis,
                preset=args.preset,
                target_height=args.target_height,
                force_scale=args.force_scale,
                voxel_size=0.03,
                remove_outliers=True,
                do_preview=args.preview,
                do_validate=not args.skip_validation,
            )
            point_cloud, meta = align_pipeline.process(
                point_cloud,
                output_path=args.save_aligned_pcd,
                preview_path=preview_file,
            )
            if args.save_aligned_pcd:
                active_pcd_path = args.save_aligned_pcd
                print(f"[Pipeline] Saved aligned point cloud: {args.save_aligned_pcd}")
            if preview_file:
                print(f"[Pipeline] Saved alignment preview: {preview_file}")
        elif not args.skip_validation:
            is_valid, report = PointCloudValidator.validate(point_cloud)
            if not is_valid or report.get("warnings"):
                print(f"\n[Pre-Flight Validation Notice for {os.path.basename(point_cloud_file)}]:")
                for w in report.get("warnings", []):
                    print(f"  - {w}")
                print("Tip: Pass --align or --preset arkit to automatically align the scan before inference.\n")

        grid_size = Layout.get_grid_size(num_bins)

        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # generate the layout
        layout = generate_layout(
            model,
            input_pcd,
            tokenizer,
            args.code_template_file,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=args.num_beams,
            seed=args.seed,
            detect_type=args.detect_type,
            categories=args.category,
        )
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # check if the output path is a file or directory
        if os.path.splitext(args.output)[-1]:
            saved_layout_path = args.output
            with open(saved_layout_path, "w") as f:
                f.write(pred_language_string)
        else:
            output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
            os.makedirs(args.output, exist_ok=True)
            saved_layout_path = os.path.join(args.output, output_filename)
            with open(saved_layout_path, "w") as f:
                f.write(pred_language_string)

        # ----------------------------------------------------
        # 2. Integrated 2D Architectural Blueprint Floor Plan
        # ----------------------------------------------------
        if args.render_floorplan:
            if os.path.splitext(args.output)[-1] or os.path.splitext(args.render_floorplan)[-1]:
                target_floorplan_path = args.render_floorplan
            else:
                os.makedirs(args.render_floorplan, exist_ok=True)
                floorplan_filename = os.path.basename(point_cloud_file).replace(".ply", "_floorplan.png")
                target_floorplan_path = os.path.join(args.render_floorplan, floorplan_filename)

            print(f"\n[Pipeline] Rendering 2D architectural blueprint floor plan...")
            render_floorplan(
                layout_path=saved_layout_path,
                pcd_path=active_pcd_path,
                output_path=target_floorplan_path,
            )
            print(f"[Pipeline] Architectural floor plan saved to: {target_floorplan_path}")
