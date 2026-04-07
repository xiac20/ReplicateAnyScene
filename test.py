import os
os.environ["LIDRA_SKIP_INIT"] = "true"
import torch
import cv2
import numpy as np
import json
import sys
import argparse


from src.models import load_vggt_model, load_sam3_image_model, load_sam3_video_model, unload_model
from src.utils import load_video_frames, vis_instance_masks
from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions, get_optimal_view_frame_id
from src.vggt_predict import vggt_predict
from src.object_segmentation import segment_wall_and_floor, segment_and_track
from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate
from src.subprocess_stages import run_sam3d_generation_stage_in_subprocess

# def main(args):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Stage 1: Progressive object discovery
#     # This part of the code is not publicly available for now. You can refer to ./assets/example/hallway.json to manually specify object categories for the scene.
#     with open(args.category_path, 'r') as f:
#         categories_and_relations = json.load(f)
#     detected_categories = list(categories_and_relations.keys())
#     print(f"Detected categories: {detected_categories}")

#     # Stage 2: Spatial-Guided Visual Deduplication
#     # Use vggt to predict 3d attributes
#     frames = load_video_frames(args.input_video, args.max_frames).to(device)
#     print(f"Loaded {len(frames)} frames for processing.")
#     vggt_model = load_vggt_model().to(device)
#     vggt_prediction_results = vggt_predict(frames, vggt_model)
#     unload_model(vggt_model)

#     # Use sam3 to predict floor and wall to align the scene to room_coordinate_system
#     sam3_image_model = load_sam3_image_model()
#     wall_masks, floor_masks = segment_wall_and_floor(vggt_prediction_results['colors'], sam3_image_model)
#     R, t = align_to_room_coordinate_system(vggt_prediction_results['world_points'], wall_masks, floor_masks)
#     vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)
#     # save related results
#     os.makedirs(os.path.join(args.output_path, 'color'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_path, 'depth'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_path, 'extrinsics'), exist_ok=True)
#     np.savetxt(os.path.join(args.output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
#     vggt_prediction_results['point_cloud_data'].export(os.path.join(args.output_path, 'point_cloud.ply'))
#     for i, image in enumerate(vggt_prediction_results['colors']):
#         cv2.imwrite(os.path.join(args.output_path, 'color', f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#     for i, depth in enumerate(vggt_prediction_results['depths']):
#         cv2.imwrite(os.path.join(args.output_path, 'depth', f"{i}.png"), (depth * 1000).astype(np.uint16)) # scale 1000
#     for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
#         np.savetxt(os.path.join(args.output_path, 'extrinsics', f"{i}.txt"), extrinsic)
#     unload_model(sam3_image_model)
#     # load sam3 video model and video frames
#     sam3_video_model = load_sam3_video_model()
#     video_path = os.path.join(args.output_path, 'color')
#     response = sam3_video_model.handle_request(
#         request=dict(
#             type="start_session",
#             resource_path=video_path,
#         )
#     )
#     session_id = response["session_id"]

#     # for each category, segment and track the instances in the video
#     all_masks = {}
#     for category in detected_categories:
#         print(f"Segmenting and tracking category: {category}")
#         category_masks = segment_and_track(category, sam3_video_model, session_id)
#         # here we deduplicate inside each category
#         deduplicated_category_masks = self_category_deduplicate(category_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
#         all_masks[category] = deduplicated_category_masks
#     # here we deduplicate inside all objects
#     deduplicated_all_masks = cross_category_deduplicate(all_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
    
#     # vis the duplicated results
#     vis_instance_masks(vggt_prediction_results['colors'], deduplicated_all_masks, os.path.join(args.output_path, 'instance_masks.mp4'))
#     unload_model(sam3_video_model)

#     # stage 3: Optimal-view asset generation
#     # get optimal view frame id for each instance
#     all_optimal_frame_ids = {}
#     for category, category_masks in deduplicated_all_masks.items():
#         all_optimal_frame_ids[category] = []
#         for instance_masks in category_masks:
#             optimal_frame_id = get_optimal_view_frame_id(vggt_prediction_results['world_points'], instance_masks)
#             all_optimal_frame_ids[category].append(optimal_frame_id)

#     # generate 3d assets for each instance using sam3d
#     inference = Inference(config_file="./models/SAM3D/checkpoints/pipeline.yaml", compile=False)
#     all_instances = {}
#     for category, category_masks in deduplicated_all_masks.items():
#         all_instances[category] = []
#         for instance_masks, optimal_frame_id in zip(category_masks, all_optimal_frame_ids[category]):
#             image = vggt_prediction_results['colors'][optimal_frame_id]
#             mask = next(im["mask"] for im in instance_masks if im["frame_id"] == optimal_frame_id)
#             pointmap = vggt_prediction_results['world_points'][optimal_frame_id]
#             extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
#             print(f"Generating 3D asset for category: {category}, optimal frame id: {optimal_frame_id}")
#             print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Pointmap shape: {pointmap.shape}, Extrinsic shape: {extrinsic.shape}")
#             print(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}, Pointmap dtype: {pointmap.dtype}, Extrinsic dtype: {extrinsic.dtype}")
#             print(f"pixel num in mask: {(mask > 0).sum()}")
#             try:
#                 instance_result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
#             except Exception as e:
#                 print(f"Error occurred while generating 3D asset for category: {category}, optimal frame id: {optimal_frame_id}")
#                 print(f"Error: {e}")
#                 continue
#             all_instances[category].append(instance_result)

#     # stage 4: Iterative Visual-Spatial Alignment
#     # This part of the code is not publicly available for now.
#     all_aligned_instances = all_instances

    # stage 5: Semantic-Aware Scene Refinement


if __name__ == "__main__":
    pointmap = np.load('/home/xc/dmy_workspace/ReplicateAnyScene/results/test/pointmap.npy')
    image = np.load('/home/xc/dmy_workspace/ReplicateAnyScene/results/test/image.npy')
    mask = np.load('/home/xc/dmy_workspace/ReplicateAnyScene/results/test/mask.npy')
    extrinsic = np.load('/home/xc/dmy_workspace/ReplicateAnyScene/results/test/extrinsic.npy')
    deduplicated_all_masks = {"debug": [[{"frame_id": 0, "mask": mask}]]}
    all_optimal_frame_ids = {"debug": [0]}
    colors = np.asarray([image])
    world_points = np.asarray([pointmap])
    extrinsics = np.asarray([extrinsic])
    output = run_sam3d_generation_stage_in_subprocess(
        deduplicated_all_masks,
        all_optimal_frame_ids,
        colors,
        world_points,
        extrinsics,
    )