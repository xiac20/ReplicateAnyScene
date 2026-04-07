import sys
import numpy as np
import pdb
import argparse
import os
sys.path.append("sam-3d-objects/notebook")
from inference import Inference, load_image, load_single_mask

import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
import trimesh

def sam3d_object_mesh(image, mask, pointmap, extrinsic):
    H, W = pointmap.shape[:2]
    points_world_flat = pointmap.reshape(-1, 3)  # shape: (H*W, 3)
    ones = np.ones((points_world_flat.shape[0], 1))
    points_world_hom = np.hstack([points_world_flat, ones])  # (N, 4)
    # 适配vggt 跟 sam3d 的诡异调整
    points_cam_hom = (np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]) @ extrinsic @ points_world_hom.T).T  # (N, 4)
    points_cam_flat = points_cam_hom[:, :3]  # 直接取前三维（因为最后一维是1）
    point_map_camera = torch.from_numpy(points_cam_flat).reshape(H, W, 3).to(torch.float32)
    print(f"point_map_camera shape: {point_map_camera.shape}, dtype: {point_map_camera.dtype}")
    print(f"image shape: {image.shape}, dtype: {image.dtype}")
    print(f"mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    original_mesh = output["glb"]
    transformed_mesh = deepcopy(original_mesh)
    # 获得变换矩阵
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    matrix_l2c = l2c_transform.get_matrix()[0].transpose(0, 1).detach().cpu().numpy()
    matrix_y2z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    matrix_adjust = np.diag([-1, -1, 1, 1])
    matrix_ext_inv = np.linalg.inv(extrinsic)
    final_transform = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z

    transformed_mesh.apply_transform(final_transform)
    return {
        "original_mesh": original_mesh,
        "transformed_mesh": transformed_mesh,
        "transform_matrix": final_transform
    }


parser = argparse.ArgumentParser(description="3D Object Reconstruction from Image and Mask")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing color images and segmentation results")
args = parser.parse_args()
input_folder = args.input_folder
color_images_dir = os.path.join(input_folder, "color")
mask_images_dir = os.path.join(input_folder, "segmentation_results_merged")
pointmap_dir = os.path.join(input_folder, "world_points")
if not os.path.exists(mask_images_dir):
    raise ValueError(f"Mask directory {mask_images_dir} does not exist.")
config_path = "./models/SAM3D/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)
objects = os.listdir(mask_images_dir)
outputs = []
for obj in objects:
    ids = os.listdir(os.path.join(mask_images_dir, obj))
    os.makedirs(os.path.join(input_folder, "object_meshes_merged", obj), exist_ok=True)
    for id_maxarea in ids:
        if id_maxarea == 'tracking_video':
            continue
        id, max_area_id = id_maxarea.split('_')
        max_area_id = int(max_area_id.replace('maxareaid', ''))
        # if not id.isdigit():
        #     continue
        os.makedirs(os.path.join(input_folder, "object_meshes_merged", obj, id_maxarea), exist_ok=True)
        # index = int(id)
        # print(f"Processing object: {obj}, index: {index}")
        # with open(os.path.join(mask_images_dir, obj, id, 'max_area_ID.txt'), 'r') as f:
        #     max_area_id = int(f.read().strip())
        image = load_image(f"{color_images_dir}/{max_area_id}.jpg")
        mask = load_single_mask(f"{mask_images_dir}/{obj}/{id_maxarea}", index=max_area_id)
        pointmap = np.load(os.path.join(pointmap_dir, f"{max_area_id}.npy"))
        extrinsics = np.loadtxt(os.path.join(input_folder, "extrinsics", f"{max_area_id}.txt"))
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Pointmap shape: {pointmap.shape}, Extrinsic shape: {extrinsics.shape}")
        print(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}, Pointmap dtype: {pointmap.dtype}, Extrinsic dtype: {extrinsics.dtype}")
        print(f"pixel num in mask: {(mask > 0).sum()}")

        results = sam3d_object_mesh(image, mask, pointmap, extrinsics)
        original_mesh = results["original_mesh"]
        transformed_mesh = results["transformed_mesh"]
        transform_matrix = results["transform_matrix"]
        original_mesh.export(os.path.join(input_folder, "object_meshes_merged", obj, id_maxarea, "original_mesh.glb"))
        transformed_mesh.export(os.path.join(input_folder, "object_meshes_merged", obj, id_maxarea, "transformed_mesh.glb"))
        np.savetxt(os.path.join(input_folder, "object_meshes_merged", obj, id_maxarea, "transform_matrix.txt"), transform_matrix)
        outputs.append(transformed_mesh)
