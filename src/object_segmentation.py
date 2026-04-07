from html import parser
import numpy as np
from PIL import Image

def segment_wall_and_floor(images, sam3_image_model):
    """
    Use SAM3 to segment wall and floor from the input images.
    
    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: The loaded SAM3 image model object.
    Returns:
        wall_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the wall).
        floor_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the floor).
        [
        
            {
                'frame_id': int,
                'mask': numpy array of shape (H, W) with binary values (True or False)
            },
            ...
        ]
    """

    wall_masks = []
    floor_masks = []
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        inference_state = sam3_image_model.set_image(image)
        sam3_image_model.reset_all_prompts(inference_state)
        inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt="single wall")
        masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            if np.sum(mask) > 500: # Filter out small masks.
                wall_masks.append({
                    'frame_id': i,
                    'mask': mask[0] # Remove the extra dimension.
                })
        sam3_image_model.reset_all_prompts(inference_state)
        inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt="floor")
        masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            if np.sum(mask) > 500: # Filter out small masks.
                floor_masks.append({
                    'frame_id': i,
                    'mask': mask[0] # Remove the extra dimension.
                })
    return wall_masks, floor_masks

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def segment_and_track(category, video_predictor, session_id):
    '''
    Segment raw instance masks using sam3 video tracking
    Args:
        category: A str means the category to segment
        video_predictor: the loaded sam3 video model  
        session_id: the session with loaded video frames corresponding to the video_predictor
    Returns:
        A list of list, each list represents a segmented instance and is composed
        of dicts with keys frame_id and mask (binary mask of the instance in that frame)
        [
            [
                {
                    'frame_id': int,
                    'mask': numpy array of shape (H, W) with binary values (True or False)
                },
                ...
            ],
            ...
        ]
    '''
    # Reset session and add text prompt for the category to segment
    _ = video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    video_predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=category))
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    if not outputs_per_frame:
        return []

    # Collect all object IDs across frames, discontinuous segments will be split into different instances.
    all_obj_ids = set()
    for frame_idx in outputs_per_frame.keys():
        all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])
    if len(all_obj_ids) == 0:
        print(f'No object detected for {category}.')
        return []
    final_results = []
    sorted_obj_ids = sorted(list(all_obj_ids))
    for obj_id in sorted_obj_ids:
        raw_frame_ids = sorted([
            id for id in outputs_per_frame.keys() 
            if obj_id in outputs_per_frame[id]['out_obj_ids']
        ])
        
        if not raw_frame_ids:
            continue
        segments = []
        if len(raw_frame_ids) > 0:
            current_segment = [raw_frame_ids[0]]
            for i in range(1, len(raw_frame_ids)):
                if raw_frame_ids[i] == raw_frame_ids[i-1] + 1:
                    current_segment.append(raw_frame_ids[i])
                else:
                    segments.append(current_segment)
                    current_segment = [raw_frame_ids[i]]
            segments.append(current_segment)
        for frame_ids in segments:
            instance_track = []
            
            for frame_id in frame_ids:
                obj_indices = np.where(outputs_per_frame[frame_id]['out_obj_ids'] == obj_id)[0]
                
                if len(obj_indices) > 0:
                    idx = obj_indices[0]
                    raw_mask = outputs_per_frame[frame_id]['out_binary_masks'][idx].squeeze()
                    binary_mask = raw_mask > 0
                    
                    instance_track.append({
                        'frame_id': frame_id,
                        'mask': binary_mask
                    })
            if instance_track:
                final_results.append(instance_track)

    return final_results