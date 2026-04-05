import argparse
import os
import torch
import cv2
import numpy as np
import json

from src.models import load_vggt_model, load_sam3_image_model, load_sam3_video_model, unload_model
from src.utils import load_video_frames, vis_instance_masks
from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions, compute_surface_area_from_pointmap
from src.vggt_predict import vggt_predict
from src.object_segmentation import segment_wall_and_floor, segment_and_track
from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Stage 1: Progressive object discovery
    # This part of the code is not publicly available for now. You can refer to ./assets/example/hallway.json to manually specify object categories for the scene.
    with open(args.category_path, 'r') as f:
        categories_and_relations = json.load(f)
    detected_categories = list(categories_and_relations.keys())
    print(f"Detected categories: {detected_categories}")

    # Stage 2: Spatial-Guided Visual Deduplication
    # Use vggt to predict 3d attributes
    frames = load_video_frames(args.input_video, args.max_frames).to(device)
    print(f"Loaded {len(frames)} frames for processing.")
    vggt_model = load_vggt_model().to(device)
    vggt_prediction_results = vggt_predict(frames, vggt_model)
    unload_model(vggt_model)

    # Use sam3 to predict floor and wall to align the scene to room_coordinate_system
    sam3_image_model = load_sam3_image_model()
    wall_masks, floor_masks = segment_wall_and_floor(vggt_prediction_results['colors'], sam3_image_model)
    R, t = align_to_room_coordinate_system(vggt_prediction_results['world_points'], wall_masks, floor_masks)
    vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)
    # save related results
    os.makedirs(os.path.join(args.output_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'extrinsics'), exist_ok=True)
    np.savetxt(os.path.join(args.output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
    vggt_prediction_results['point_cloud_data'].export(os.path.join(args.output_path, 'point_cloud.ply'))
    for i, image in enumerate(vggt_prediction_results['colors']):
        cv2.imwrite(os.path.join(args.output_path, 'color', f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for i, depth in enumerate(vggt_prediction_results['depths']):
        cv2.imwrite(os.path.join(args.output_path, 'depth', f"{i}.png"), (depth * 1000).astype(np.uint16)) # scale 1000
    for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
        np.savetxt(os.path.join(args.output_path, 'extrinsics', f"{i}.txt"), extrinsic)
    unload_model(sam3_image_model)
    # load sam3 video model and video frames
    sam3_video_model = load_sam3_video_model()
    video_path = os.path.join(args.output_path, 'color')
    response = sam3_video_model.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    # for each category, segment and track the instances in the video
    all_masks = {}
    for category in detected_categories:
        print(f"Segmenting and tracking category: {category}")
        category_masks = segment_and_track(category, sam3_video_model, session_id)
        # here we deduplicate inside each category
        deduplicated_category_masks = self_category_deduplicate(category_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
        all_masks[category] = deduplicated_category_masks
    # here we deduplicate inside all objects
    deduplicated_all_masks = cross_category_deduplicate(all_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
    
    # vis the duplicated results
    vis_instance_masks(vggt_prediction_results['colors'], deduplicated_all_masks, os.path.join(args.output_path, 'instance_masks.mp4'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReplicateAnyScene pipeline")
    parser.add_argument("--input_video", type=str, default='./assets/example/hallway.mp4', help="Path to input video file or directory of images")
    parser.add_argument("--output_path", type=str, default='./results/hallway', help="Directory to save output results")
    parser.add_argument("--category_path", type=str, default='./assets/example/hallway.json', help="path to category and relation json")
    parser.add_argument("--max_frames", type=int, default=160, help="Maximum number of frames to process from the video")
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"Input video or image directory not found: {args.input_video}")
    os.makedirs(args.output_path, exist_ok=True)
    main(args)