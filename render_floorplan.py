#!/usr/bin/env python3
"""
Architectural 2D Floor Plan Renderer for SpatialLM.

Parses SpatialLM predicted layout (walls, doors, windows, and oriented 3D bounding boxes)
and generates high-resolution, uncropped 2D architectural blueprint floor plans with:
- Solid architectural walls
- Oriented colored furniture bounding boxes with category labels
- Door swings and window indicators
- True room extent metrics and scale grid
- Guaranteed aspect-ratio preservation without axis cropping
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors


def render_floorplan(layout_path, pcd_path=None, output_path="floorplan.png", dpi=300):
    if not os.path.exists(layout_path):
        raise FileNotFoundError(f"Layout file not found: {layout_path}")

    with open(layout_path, "r") as f:
        layout_text = f.read()

    layout = Layout(layout_text)

    # ----------------------------------------------------
    # Collect All Coordinates for Uncropped Dynamic Sizing
    # ----------------------------------------------------
    all_x = []
    all_y = []

    # Wall coordinates
    for wall in layout.walls:
        all_x.extend([wall.ax, wall.bx])
        all_y.extend([wall.ay, wall.by])

    # Windows and Doors
    for win in layout.windows:
        all_x.append(win.position_x)
        all_y.append(win.position_y)
    for door in layout.doors:
        all_x.append(door.position_x)
        all_y.append(door.position_y)

    # Oriented Bbox corners (exact trigonometric projection)
    for bbox in layout.bboxes:
        cx, cy = bbox.position_x, bbox.position_y
        w, l = bbox.scale_x, bbox.scale_y
        yaw = bbox.angle_z  # in radians

        dx = w / 2.0
        dy = l / 2.0
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # 4 rotated corner coordinates
        corners_x = [
            cx + dx * cos_y - dy * sin_y,
            cx - dx * cos_y - dy * sin_y,
            cx - dx * cos_y + dy * sin_y,
            cx + dx * cos_y + dy * sin_y,
        ]
        corners_y = [
            cy + dx * sin_y + dy * cos_y,
            cy - dx * sin_y + dy * cos_y,
            cy - dx * sin_y - dy * cos_y,
            cy + dx * sin_y - dy * cos_y,
        ]
        all_x.extend(corners_x)
        all_y.extend(corners_y)

    # Optional: point cloud points
    points = None
    if pcd_path and os.path.exists(pcd_path):
        pcd = load_o3d_pcd(pcd_path)
        points, _ = get_points_and_colors(pcd)
        if len(points) > 0:
            # Include 99% robust bounds of point cloud so points fit inside
            all_x.extend([np.percentile(points[:, 0], 1), np.percentile(points[:, 0], 99)])
            all_y.extend([np.percentile(points[:, 1], 1), np.percentile(points[:, 1], 99)])

    if not all_x or not all_y:
        all_x = [0, 5]
        all_y = [0, 5]

    span_x = max(all_x) - min(all_x)
    span_y = max(all_y) - min(all_y)
    aspect_ratio = span_x / max(span_y, 0.1)

    # ----------------------------------------------------
    # Dynamic Canvas Sizing (Prevents Cropping)
    # ----------------------------------------------------
    fig_w = 14.0
    fig_h = max(7.0, min(16.0, fig_w / max(aspect_ratio, 0.4)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    # Draw point cloud background if available
    if points is not None and len(points) > 0:
        if len(points) > 50000:
            sub_idx = np.random.choice(len(points), 50000, replace=False)
            sampled_points = points[sub_idx]
        else:
            sampled_points = points

        ax.scatter(
            sampled_points[:, 0],
            sampled_points[:, 1],
            s=0.5,
            c="#d0d5dd",
            alpha=0.35,
            label="Point Cloud Scan",
            zorder=1,
        )

    # ----------------------------------------------------
    # 1. Draw Architectural Walls
    # ----------------------------------------------------
    for wall in layout.walls:
        ax.plot(
            [wall.ax, wall.bx],
            [wall.ay, wall.by],
            color="#101828",
            linewidth=4.5,
            solid_capstyle="round",
            zorder=3,
        )

    # ----------------------------------------------------
    # 2. Draw Windows & Doors
    # ----------------------------------------------------
    for win in layout.windows:
        ax.plot(
            win.position_x,
            win.position_y,
            marker="s",
            color="#0ba5ec",
            markersize=7,
            zorder=4,
        )

    for door in layout.doors:
        ax.plot(
            door.position_x,
            door.position_y,
            marker="o",
            color="#f79009",
            markersize=8,
            zorder=4,
        )

    # ----------------------------------------------------
    # 3. Draw Oriented Furniture Bboxes
    # ----------------------------------------------------
    color_palette = {
        "bed": "#7a5af8",
        "sofa": "#2e90fa",
        "combination_sofa": "#2e90fa",
        "chair": "#12b76a",
        "dining_chair": "#12b76a",
        "table": "#f79009",
        "dining_table": "#f79009",
        "coffee_table": "#f79009",
        "nightstand": "#ee46bc",
        "wardrobe": "#475467",
        "cabinet": "#475467",
        "tv_cabinet": "#475467",
        "bathroom_cabinet": "#0e7090",
        "refrigerator": "#0891b2",
        "stool": "#16a34a",
        "carpet": "#98a2b3",
        "painting": "#ec4899",
        "curtain": "#6366f1",
        "plants": "#059669",
    }

    for bbox in layout.bboxes:
        cx, cy = bbox.position_x, bbox.position_y
        w, l = bbox.scale_x, bbox.scale_y
        yaw = bbox.angle_z  # in radians

        # Color lookup
        cat = bbox.class_name.lower()
        color = "#344054"
        for key in color_palette:
            if key in cat:
                color = color_palette[key]
                break

        # Oriented rectangle
        rect = patches.Rectangle(
            (-w / 2.0, -l / 2.0),
            w,
            l,
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.35,
            zorder=5,
        )
        t = (
            Affine2D()
            .rotate(yaw)
            .translate(cx, cy)
            + ax.transData
        )
        rect.set_transform(t)
        ax.add_patch(rect)

        # Label inside bbox
        deg_rot = np.degrees(yaw)
        if abs(deg_rot) > 90:
            deg_rot = (deg_rot + 180) % 360 - 180

        ax.text(
            cx,
            cy,
            bbox.class_name,
            color="#101828",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            rotation=deg_rot,
            zorder=6,
        )

    # ----------------------------------------------------
    # 4. Strict Non-Cropping Bounds & Aspect Equalization
    # ----------------------------------------------------
    # Add generous 12% padding around the maximum extents
    pad_x = max(span_x * 0.12, 0.8)
    pad_y = max(span_y * 0.12, 0.8)

    min_x_view = min(all_x) - pad_x
    max_x_view = max(all_x) + pad_x
    min_y_view = min(all_y) - pad_y
    max_y_view = max(all_y) + pad_y

    ax.set_xlim(min_x_view, max_x_view)
    ax.set_ylim(min_y_view, max_y_view)

    # Use adjustable='box' to prevent matplotlib from overriding/cropping data limits
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35, color="#98a2b3")

    # Title with true room measurements
    wall_x = [w.ax for w in layout.walls] + [w.bx for w in layout.walls]
    wall_y = [w.ay for w in layout.walls] + [w.by for w in layout.walls]
    if wall_x and wall_y:
        actual_room_w = max(wall_x) - min(wall_x)
        actual_room_l = max(wall_y) - min(wall_y)
        title_text = f"SpatialLM Generated Architectural Floor Plan\nRoom Extent: {actual_room_w:.2f}m (Width) × {actual_room_l:.2f}m (Length)"
    else:
        title_text = f"SpatialLM Generated Architectural Floor Plan\nExtent: {span_x:.2f}m × {span_y:.2f}m"

    ax.set_title(title_text, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)

    # Legend
    legend_elements = [
        patches.Patch(color="#101828", label="Wall"),
        patches.Patch(color="#f79009", label="Door"),
        patches.Patch(color="#0ba5ec", label="Window"),
        patches.Patch(color="#7a5af8", alpha=0.6, label="Furniture (Bbox)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.92)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Floor plan successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate 2D Floor Plan PNG from SpatialLM layout")
    parser.add_argument("-l", "--layout", type=str, required=True, help="Path to predicted_layout.txt")
    parser.add_argument("-p", "--point_cloud", type=str, default=None, help="Optional point cloud .ply for overlay")
    parser.add_argument("-o", "--output", type=str, default="floorplan.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI resolution")
    args = parser.parse_args()

    render_floorplan(args.layout, args.point_cloud, args.output, args.dpi)
