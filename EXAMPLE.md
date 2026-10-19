# Example using a custom video

A new open-source SLAM system [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) just released. We found it is great for reconstructing point clouds from RGB inputs as well. Here we provide an example of how to use it to reconstruct a point cloud from a video, and use our model to estimate the layout of the scene. Following steps work for MASt3R-SLAM, and other reconstruction methods as well.

## 1. Reconstruct a point cloud with SLAM3R

Please refer to the [SLAM3R Replica Example](https://github.com/PKU-VCL-3DV/SLAM3R?tab=readme-ov-file#replica-dataset) for the details of how to run SLAM3R on the provided Replica example.

You can set `sample_freq` in this [line](https://github.com/PKU-VCL-3DV/SLAM3R/blob/3831478e15b83509f3f317abc1d4f7bd3cc21e2d/scripts/demo_replica.sh#L8) to 2 to decrease the GPU memory usage and speed up the reconstruction. And you can also try increase the `CONF_THRES_L2W` threshold [here](https://github.com/PKU-VCL-3DV/SLAM3R/blob/3831478e15b83509f3f317abc1d4f7bd3cc21e2d/scripts/demo_replica.sh#L22) to get cleaner point cloud.

If you are using MASt3R-SLAM, we recommend you to increase the `C_conf_threshold` and add some point cloud cleaning steps for each keyframe point cloud exportation in [line](https://github.com/rmurai0610/MASt3R-SLAM/blob/c3d0d5b67bf51d558d7640ff6032407f68041f92/mast3r_slam/evaluate.py#L65), such as removing statistical outliers with following code. These will generally help to get cleaner point clouds.

```python
...
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_pts)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)
pcd, trace = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
points = np.asarray(pcd.points)
color = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
pointclouds.append(points)
colors.append(color)
```

## 2. All-in-One End-to-End Pipeline (Alignment + Inference + Floor Plan)

Instead of manually running separate alignment, validation, and rendering scripts, **`inference.py` now integrates the entire workflow into a single command**:

```bash
python inference.py \
    --point_cloud raw_slam_scan.ply \
    --output predicted_layout.txt \
    --model_path manycore-research/SpatialLM-Llama-1B \
    --align \
    --preset arkit \
    --target_height 2.65 \
    --preview \
    --save_aligned_pcd scene_aligned.ply \
    --render_floorplan floorplan.png
```

### What this single command does automatically:
1. **Automated Manhattan Alignment & Scale Calibration**: Leverages our official [3D-PointCloud-Aligner](https://github.com/A511-git/3D-PointCloud-Aligner) to orient the vertical axis (+Z Up), square walls to the Manhattan frame (X/Y axes) via 4-fold periodic normal voting, ground the floor at Z=0, and calibrate metric room scale.
2. **Diagnostic Previews**: Automatically saves a 3-view orthogonal projection PNG (`scene_aligned_preview.png`) for instant verification.
3. **Pre-Flight Validation**: Asserts point count, bounding spans, and floor level.
4. **SpatialLM Inference**: Predicts walls, doors, windows, and 3D oriented furniture bounding boxes.
5. **2D Architectural Blueprint Rendering**: Generates a high-resolution, uncropped 2D floor plan PNG (`floorplan.png`).

|                     Before Alignment                     |                    After Alignment                    |
| :------------------------------------------------------: | :---------------------------------------------------: |
| ![Point cloud before alignment](./figures/unaligned.jpg) | ![Point cloud after alignment](./figures/aligned.jpg) |

## 3. Visualizing 3D Layout (Optional)

You can visualize the predicted 3D bounding boxes interactively using Rerun:

```bash
# visualize the layout interactively
python visualize.py --point_cloud scene_aligned.ply --layout predicted_layout.txt --save replica_layout.rrd
rerun replica_layout.rrd
```

|                      View 1                      |                      View 2                      |
| :----------------------------------------------: | :----------------------------------------------: |
| ![Result view 1](./figures/replica_example1.jpg) | ![Result view 2](./figures/replica_example2.jpg) |

## 5. Summary

In this example, we demonstrated how to estimate scene layout from a custom video. The workflow involves four main steps: reconstructing a point cloud from video, aligning it to match the ScanNet orientation convention, scale the point cloud to a roughly metric accurate scale, and finally running our layout estimation model. We will keep improving to make these steps more automated, efficient and robust.

Note that in this example, one sofa is not detected in the prediction. This is because our training data primarily consists of Chinese residential apartments where living rooms typically feature a single main sofa facing a TV, which is on the other side of the room. The model currently has its limitation on generalize very well to more diverse scenarios, such as non-standard layouts, cluttered spaces, or outdoor environments. We are actively working on it, and will try to improve the model's performance to more diverse scenes in the next release.
