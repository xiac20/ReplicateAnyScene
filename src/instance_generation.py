import numpy as np
import torch
import os
import multiprocessing as mp
import importlib.util
import traceback
from queue import Empty
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

def generate_3d_asset(image, mask, pointmap, extrinsic, inference):
    '''
    3D asset generation for a single object instance using sam3d. 
    Args:
        image: (H, W, 3) numpy array, RGB image of the scene
        mask: (H, W) numpy array, binary mask of the object instance
        pointmap: (H, W, 3) numpy array, world coordinates of each pixel in the image
        extrinsic: (4, 4) numpy array, camera extrinsic matrix for the image
        inference: sam3d inference object
    Returns:    A dictionary containing:
        - mesh: A trimesh object representing the reconstructed 3D mesh of the object instance in object coordinate system
        - T: the transformation matrix to align the mesh to the world coordinate system
    '''
    H, W = pointmap.shape[:2]
    points_world_flat = pointmap.reshape(-1, 3)  # shape: (H*W, 3)
    ones = np.ones((points_world_flat.shape[0], 1))
    points_world_hom = np.hstack([points_world_flat, ones])  # (N, 4)

    # vggt format to sam3d format
    points_cam_hom = (np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]) @ extrinsic @ points_world_hom.T).T  # (N, 4)
    points_cam_flat = points_cam_hom[:, :3]
    point_map_camera = torch.from_numpy(points_cam_flat).reshape(H, W, 3).to(torch.float32)
    point_map_camera = point_map_camera.contiguous()
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    original_mesh = output["glb"]

    # get the transformation matrix. The transformation is z-up.
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

    return {
        "original_mesh": original_mesh,
        "T": final_transform
    }


def _generate_3d_asset_worker(queue, image, mask, pointmap, extrinsic, config_file, compile_model):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference

        inference = Inference(config_file=config_file, compile=compile_model)
        result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
        queue.put((True, result))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _generate_all_instances_worker(
    queue,
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file,
    compile_model,
):
    try:
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        inference_file = repo_root / "sam-3d-objects" / "notebook" / "inference.py"
        spec = importlib.util.spec_from_file_location("sam3d_notebook_inference", inference_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load inference module from {inference_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Inference = module.Inference

        inference = Inference(config_file=config_file, compile=compile_model)
        all_instances = {}
        for category, category_masks in deduplicated_all_masks.items():
            all_instances[category] = []
            for instance_masks, optimal_frame_id in zip(category_masks, all_optimal_frame_ids[category]):
                image = colors[optimal_frame_id]
                mask = next(im["mask"] for im in instance_masks if im["frame_id"] == optimal_frame_id)
                pointmap = world_points[optimal_frame_id]
                extrinsic = extrinsics[optimal_frame_id]
                print(
                    f"[SAM3D subprocess] category={category}, frame={optimal_frame_id}, "
                    f"mask_sum={int(np.asarray(mask).sum())}"
                )
                instance_result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
                all_instances[category].append(instance_result)
        print("[SAM3D subprocess] finished generating all instances, putting results in the queue. This may take a while...")
        queue.put((True, all_instances))
    except Exception:
        queue.put((False, traceback.format_exc()))
    finally:
        queue.close()
        queue.cancel_join_thread()


def generate_3d_asset_in_subprocess(
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file="./models/SAM3D/checkpoints/pipeline.yaml",
    compile_model=False,
):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_generate_all_instances_worker,
        args=(
            queue,
            deduplicated_all_masks,
            all_optimal_frame_ids,
            colors,
            world_points,
            extrinsics,
            config_file,
            compile_model,
        ),
    )
    process.start()
    try:
        try:
            # Read the payload before join(); joining first can deadlock on Queue finalizers.
            ok, payload = queue.get(timeout=7200)
        except Empty:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            raise RuntimeError("subprocess timed out waiting for result")

        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

        if ok:
            return payload
        raise RuntimeError(payload)
    finally:
        queue.close()
        queue.cancel_join_thread()
