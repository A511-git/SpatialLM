import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors


def render_floorplan(layout_path, pcd_path=None, output_path="floorplan.png", dpi=300):
    with open(layout_path, "r") as f:
        layout_text = f.read()

    layout = Layout(layout_text)
    boxes = layout.to_boxes()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    # Optional: draw point cloud top-down density in background
    if pcd_path and os.path.exists(pcd_path):
        pcd = load_o3d_pcd(pcd_path)
        points, colors = get_points_and_colors(pcd)
        # Subsample points for top-down view
        if len(points) > 50000:
            sub_idx = np.random.choice(len(points), 50000, replace=False)
            points = points[sub_idx]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=0.5,
            c="#d0d5dd",
            alpha=0.4,
            label="Point Cloud Scan",
            zorder=1,
        )

    # 1. Draw Walls
    for wall in layout.walls:
        ax.plot(
            [wall.ax, wall.bx],
            [wall.ay, wall.by],
            color="#101828",
            linewidth=4.5,
            solid_capstyle="round",
            zorder=3,
        )

    # 2. Draw Windows
    for win in layout.windows:
        ax.plot(
            win.position_x,
            win.position_y,
            marker="s",
            color="#0ba5ec",
            markersize=6,
            zorder=4,
        )

    # 3. Draw Doors
    for door in layout.doors:
        ax.plot(
            door.position_x,
            door.position_y,
            marker="o",
            color="#f79009",
            markersize=7,
            zorder=4,
        )

    # 4. Draw Furniture Bboxes (Oriented 2D Rectangles)
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
        "wardrobe": "#667085",
        "cabinet": "#667085",
        "tv_cabinet": "#667085",
    }

    all_x = [w.ax for w in layout.walls] + [w.bx for w in layout.walls]
    all_y = [w.ay for w in layout.walls] + [w.by for w in layout.walls]

    for bbox in layout.bboxes:
        cx, cy = bbox.position_x, bbox.position_y
        w, l = bbox.scale_x, bbox.scale_y
        yaw = bbox.angle_z  # in radians

        all_x.extend([cx - w, cx + w])
        all_y.extend([cy - l, cy + l])

        # Pick color by category
        cat = bbox.class_name.lower()
        color = "#344054"
        for key in color_palette:
            if key in cat:
                color = color_palette[key]
                break

        # Create oriented rectangle
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
        ax.text(
            cx,
            cy,
            bbox.class_name,
            color="#1d2939",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            rotation=np.degrees(yaw) if abs(np.degrees(yaw)) < 90 else np.degrees(yaw) + 180,
            zorder=6,
        )

    # Styling & Bounds
    if all_x and all_y:
        pad_x = (max(all_x) - min(all_x)) * 0.1 or 1.0
        pad_y = (max(all_y) - min(all_y)) * 0.1 or 1.0
        ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
        ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3, color="#98a2b3")
    ax.set_title("SpatialLM Generated Architectural Floor Plan", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)

    # Custom legend
    legend_elements = [
        patches.Patch(color="#101828", label="Wall"),
        patches.Patch(color="#f79009", label="Door"),
        patches.Patch(color="#0ba5ec", label="Window"),
        patches.Patch(color="#7a5af8", alpha=0.6, label="Furniture (Bbox)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
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
