#!/usr/bin/env python3
"""
Automated Synthetic Test for SpatialLM Point Cloud Alignment and Validation.

Constructs a mathematically exact 3D indoor room with metric scale (5m x 4m x 2.8m),
tests validation, perturbs with rotation/tilt/noise, aligns with align_pcd.py,
and verifies 100% compliance with SpatialLM specifications.
"""

import os
import sys
import numpy as np
import open3d as o3d

from align_pcd import align_point_cloud
from validate_pcd import validate_point_cloud


def create_synthetic_room(length=5.0, width=4.0, height=2.8, density=600):
    """
    Creates a synthetic 3D point cloud of a room with floor, ceiling, 4 walls, and a bed.
    """
    pts = []
    colors = []

    # Floor (z = 0)
    fx = np.random.uniform(0, length, int(length * width * density))
    fy = np.random.uniform(0, width, len(fx))
    fz = np.zeros_like(fx)
    pts.append(np.stack([fx, fy, fz], axis=1))
    colors.append(np.full((len(fx), 3), [0.8, 0.75, 0.7]))  # Light wood floor

    # Ceiling (z = height)
    cx = np.random.uniform(0, length, int(length * width * density * 0.7))
    cy = np.random.uniform(0, width, len(cx))
    cz = np.full_like(cx, height)
    pts.append(np.stack([cx, cy, cz], axis=1))
    colors.append(np.full((len(cx), 3), [0.95, 0.95, 0.95]))  # White ceiling

    # Wall 1: X = 0 (Y-Z plane)
    w1_y = np.random.uniform(0, width, int(width * height * density))
    w1_z = np.random.uniform(0, height, len(w1_y))
    w1_x = np.zeros_like(w1_y)
    pts.append(np.stack([w1_x, w1_y, w1_z], axis=1))
    colors.append(np.full((len(w1_y), 3), [0.85, 0.85, 0.9]))

    # Wall 2: X = length (Y-Z plane)
    w2_y = np.random.uniform(0, width, int(width * height * density))
    w2_z = np.random.uniform(0, height, len(w2_y))
    w2_x = np.full_like(w2_y, length)
    pts.append(np.stack([w2_x, w2_y, w2_z], axis=1))
    colors.append(np.full((len(w2_y), 3), [0.85, 0.85, 0.9]))

    # Wall 3: Y = 0 (X-Z plane)
    w3_x = np.random.uniform(0, length, int(length * height * density))
    w3_z = np.random.uniform(0, height, len(w3_x))
    w3_y = np.zeros_like(w3_x)
    pts.append(np.stack([w3_x, w3_y, w3_z], axis=1))
    colors.append(np.full((len(w3_x), 3), [0.88, 0.88, 0.88]))

    # Wall 4: Y = width (X-Z plane)
    w4_x = np.random.uniform(0, length, int(length * height * density))
    w4_z = np.random.uniform(0, height, len(w4_x))
    w4_y = np.full_like(w4_x, width)
    pts.append(np.stack([w4_x, w4_y, w4_z], axis=1))
    colors.append(np.full((len(w4_x), 3), [0.88, 0.88, 0.88]))

    # Furniture: Bed box (2m x 1.8m x 0.6m) resting on floor
    bx = np.random.uniform(1.0, 3.0, 1500)
    by = np.random.uniform(1.0, 2.8, 1500)
    bz = np.random.uniform(0.0, 0.6, 1500)
    pts.append(np.stack([bx, by, bz], axis=1))
    colors.append(np.full((len(bx), 3), [0.3, 0.4, 0.8]))

    all_pts = np.vstack(pts)
    all_colors = np.vstack(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    return pcd


def run_pipeline_test():
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)

    gt_ply = os.path.join(test_dir, "synthetic_ground_truth.ply")
    perturbed_ply = os.path.join(test_dir, "synthetic_perturbed.ply")
    aligned_ply = os.path.join(test_dir, "synthetic_recovered_aligned.ply")

    # 1. Generate Ground Truth Metric Room
    print("\n--- Test 1: Creating Ground Truth Metric Room ---")
    pcd_gt = create_synthetic_room(length=5.0, width=4.0, height=2.8)
    o3d.io.write_point_cloud(gt_ply, pcd_gt)
    print(f"Saved ground truth room: {gt_ply}")

    # Validate GT room
    print("\nValidating Ground Truth Room:")
    gt_ready = validate_point_cloud(gt_ply)
    assert gt_ready, "Ground truth room should pass all validation checks!"

    # 2. Perturb Room (Rotate 35° yaw, rotate to Y-up ARKit convention, add floating outliers)
    print("\n--- Test 2: Creating Perturbed Scan (Tilt, Yaw, Outliers) ---")
    pts = np.asarray(pcd_gt.points).copy()

    # Add 35 deg yaw rotation
    theta = np.deg2rad(35.0)
    R_yaw = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    pts = pts @ R_yaw.T

    # Rotate +Z Up -> +Y Up (ARKit camera convention)
    # [X, Y, Z] -> [X, Z, -Y]
    R_arkit = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float64)
    pts = pts @ R_arkit.T

    # Add floating outliers 15 meters away
    outliers = np.array([
        [15.0, 12.0, 8.0],
        [-10.0, 14.0, -9.0],
        [12.0, -11.0, 10.0]
    ])
    pts = np.vstack([pts, outliers])
    colors = np.vstack([np.asarray(pcd_gt.colors), np.full((len(outliers), 3), 0.5)])

    pcd_perturbed = o3d.geometry.PointCloud()
    pcd_perturbed.points = o3d.utility.Vector3dVector(pts)
    pcd_perturbed.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(perturbed_ply, pcd_perturbed)
    print(f"Saved perturbed room: {perturbed_ply}")

    # Validate perturbed room - should detect failures
    print("\nValidating Perturbed Scan (Should flag issues):")
    pert_valid = validate_point_cloud(perturbed_ply)
    assert not pert_valid, "Perturbed scan must flag validation warnings!"

    # 3. Align with updated align_pcd.py
    print("\n--- Test 3: Running align_pcd.py with Multi-Plane Consensus & Preview ---")
    align_point_cloud(
        input_path=perturbed_ply,
        output_path=aligned_ply,
        up_axis="auto",
        target_height=2.8,
        preview=True
    )

    # 4. Re-validate Aligned Room
    print("\n--- Test 4: Validating Aligned Room Output ---")
    recovered_valid = validate_point_cloud(aligned_ply)
    assert recovered_valid, "Aligned point cloud must pass 100% of SpatialLM requirements!"

    # Check physical properties of aligned room
    pcd_aligned = o3d.io.read_point_cloud(aligned_ply)
    aligned_pts = np.asarray(pcd_aligned.points)

    recovered_height = aligned_pts[:, 2].max() - aligned_pts[:, 2].min()
    assert 2.6 <= recovered_height <= 3.0, f"Recovered height {recovered_height:.2f}m not in expected ~2.8m range!"

    print("\n" + "=" * 55)
    print("  ALL SYNTHETIC PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_pipeline_test()
