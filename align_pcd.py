#!/usr/bin/env python3
"""
SpatialLM Point Cloud Aligner CLI.
Directly powered by the 3D-PointCloud-Aligner engine:
https://github.com/A511-git/3D-PointCloud-Aligner

Converts arbitrary 3D indoor point clouds into the ScanNet orientation convention:
- Floor leveled and vertical axis aligned to +Z Up
- Walls aligned to X and Y coordinate axes (Manhattan Frame)
- Scale calibrated to metric units (1 unit = 1 meter)
- Grounded at Z = 0 and centered at (0, 0)
- Statistical outlier removal and voxel downsampling
"""

import os
import sys
import argparse

# Link to 3D-PointCloud-Aligner submodule
ALIGNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D-PointCloud-Aligner")
if ALIGNER_DIR not in sys.path:
    sys.path.insert(0, ALIGNER_DIR)

from src.pipeline import PointCloudPipeline
from src.validator import PointCloudValidator


def align_point_cloud(
    input_path: str,
    output_path: str = "scene_aligned.ply",
    up_axis: str = "auto",
    preset: str = None,
    target_height: float = 2.65,
    force_scale: bool = False,
    voxel_size: float = 0.03,
    remove_outliers: bool = True,
    preview: bool = False,
    meta_path: str = None,
):
    """
    Main programmatic interface to align point clouds using 3D-PointCloud-Aligner.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input point cloud not found: {input_path}")

    print("\n" + "=" * 60)
    print("  SpatialLM: Running 3D-PointCloud-Aligner Engine")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    pipeline = PointCloudPipeline(
        up_axis=up_axis,
        preset=preset,
        target_height=target_height,
        force_scale=force_scale,
        voxel_size=voxel_size,
        remove_outliers=remove_outliers,
        do_preview=preview,
        do_validate=True
    )

    preview_file = (os.path.splitext(output_path)[0] + "_preview.png") if preview else None
    pcd, meta = pipeline.process(
        input_path,
        output_path=output_path,
        preview_path=preview_file
    )

    if meta_path:
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved alignment metadata manifest: {meta_path}")

    return pcd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Align 3D point clouds for SpatialLM using 3D-PointCloud-Aligner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .ply/.pcd file or directory.")
    parser.add_argument("-o", "--output", default="scene_aligned.ply", help="Path to output .ply file or directory.")
    parser.add_argument("--up-axis", default="auto", choices=["auto", "x", "-x", "y", "-y", "z", "-z"], help="Vertical axis direction.")
    parser.add_argument("--preset", default=None, choices=["arkit", "opencv", "opengl", "scannet"], help="Sensor orientation preset.")
    parser.add_argument("--target-height", type=float, default=2.65, help="Target room height in meters.")
    parser.add_argument("--force-scale", action="store_true", help="Force scaling to target-height.")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Voxel size in meters for downsampling.")
    parser.add_argument("--no-clean", action="store_true", help="Skip statistical outlier removal.")
    parser.add_argument("--preview", action="store_true", help="Generate 3-view orthogonal diagnostic PNG.")
    parser.add_argument("--meta", default=None, help="Path to save metadata JSON.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if os.path.isdir(args.input):
        pipeline = PointCloudPipeline(
            up_axis=args.up_axis,
            preset=args.preset,
            target_height=args.target_height,
            force_scale=args.force_scale,
            voxel_size=args.voxel_size,
            remove_outliers=not args.no_clean,
            do_preview=args.preview,
            do_validate=True
        )
        pipeline.process_directory(args.input, args.output, generate_previews=args.preview)
    else:
        align_point_cloud(
            input_path=args.input,
            output_path=args.output,
            up_axis=args.up_axis,
            preset=args.preset,
            target_height=args.target_height,
            force_scale=args.force_scale,
            voxel_size=args.voxel_size,
            remove_outliers=not args.no_clean,
            preview=args.preview,
            meta_path=args.meta
        )


if __name__ == "__main__":
    main()
