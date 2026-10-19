#!/usr/bin/env python3
"""
SpatialLM Point Cloud Pre-Flight Validator.
Powered by 3D-PointCloud-Aligner: https://github.com/A511-git/3D-PointCloud-Aligner
"""

import os
import sys
import argparse

# Link to 3D-PointCloud-Aligner submodule
ALIGNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D-PointCloud-Aligner")
if ALIGNER_DIR not in sys.path:
    sys.path.insert(0, ALIGNER_DIR)

from src.validator import PointCloudValidator


def validate_point_cloud(file_path: str, verbose: bool = False) -> bool:
    """
    Validates a 3D point cloud against SpatialLM specifications.
    Returns True if compliant, False if issues/warnings are flagged.
    """
    if not os.path.exists(file_path):
        print(f"Error: Point cloud file not found: {file_path}")
        return False

    print("\n" + "=" * 62)
    print(f"  SpatialLM Pre-Flight Validator: {os.path.basename(file_path)}")
    print("=" * 62)

    is_valid, report = PointCloudValidator.validate(file_path)

    metrics = report.get("metrics", {})
    if metrics:
        print(f"  [METRICS] Points: {report['num_points']:,}")
        print(f"  [METRICS] Room Spans: X={metrics.get('x_span_m')}m, Y={metrics.get('y_span_m')}m, Z={metrics.get('z_span_m')}m")
        print(f"  [METRICS] Effective Height: {metrics.get('effective_height_m')}m (Floor at Z={metrics.get('floor_level_z')}m)")

    checks = report.get("checks", {})
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[WARN]"
        print(f"  {status:<8} {check_name.replace('_', ' ').title()}")

    print("\n" + "-" * 62)
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])

    if is_valid and len(warnings) == 0:
        print("  VERDICT: [READY] Point cloud meets all SpatialLM specifications!")
        print("  You can proceed directly to inference:")
        print(f"    python inference.py --point_cloud {file_path} --output pred_layout.txt")
        print("=" * 62 + "\n")
        return True
    else:
        print("  VERDICT: [ACTION REQUIRED] Point cloud needs alignment / preprocessing.")
        for idx, item in enumerate(errors + warnings, 1):
            print(f"    {idx}. {item}")
        print("\n  Recommended Command:")
        print(f"    python align_pcd.py -i {file_path} -o {os.path.splitext(file_path)[0]}_aligned.ply --preview")
        print("=" * 62 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SpatialLM Point Cloud Pre-Flight Validator"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to point cloud (.ply, .pcd)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose details")
    args = parser.parse_args()

    validate_point_cloud(args.input, args.verbose)


if __name__ == "__main__":
    main()
