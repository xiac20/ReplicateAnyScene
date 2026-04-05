from vggt.utils.load_fn import load_and_preprocess_images
import os
import re
import numpy as np
import tempfile
import colorcet as cc
import matplotlib.colors as mcolors
import cv2

def load_video_frames(video_path, max_frames):
    '''
    Load uniformly sampled frames from a video file or directory of images.
    '''
    if os.path.isdir(video_path):
        images = os.listdir(video_path)
        images = [img for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]
        images = sorted(images, key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1)
        total_frames = len(images)
        if total_frames == 0:
            raise ValueError(f"No image files found in directory: {video_path}")
        if total_frames > max_frames and max_frames > 0:
            indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
            images = [os.path.join(video_path, images[i]) for i in indices]
        else:
            images = [os.path.join(video_path, img) for img in images]
        return load_and_preprocess_images(images)
    else:
        # Use ffmpeg to extract frames from the video in a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            os.system(f"ffmpeg -i {video_path} -vsync 0 {temp_dir}/frame_%04d.png")
            return load_video_frames(temp_dir, max_frames)
        
def get_glasbey_colors(n):
    '''
        load n BGR tuples for visualization
    '''
    hex_colors = cc.glasbey[:n]
    bgr_colors = []
    for hex_val in hex_colors:
        rgb_float = mcolors.to_rgb(hex_val)
        r, g, b = tuple(int(x * 255) for x in rgb_float)
        bgr_colors.append((b, g, r))
    return bgr_colors

def vis_instance_masks(video_frames, all_masks, output_path):
    '''
        Visualize segmentation results in a video
        Args:
            video_frames: numpy array of shape (S, H, W, 3)
            all_masks: A dict returned by cross_category_deduplicate function
            output_path: path to save the video
    '''
    # to bgr format
    frames_to_show = video_frames[:,:,:,::-1].copy()

    # assign each instance a color for vis
    instance_num = 0
    for category, category_masks in all_masks.items():
        instance_num = instance_num + len(category_masks)
    colors = get_glasbey_colors(instance_num)

    # color each instance in the video frames
    color_idx = 0
    for category, category_masks in all_masks.items():
        for instance_id, instance_masks in enumerate(category_masks):
            color = colors[color_idx]
            for instance_mask in instance_masks:
                frame_id = instance_mask['frame_id']
                mask = instance_mask['mask']

                # add color to the mask
                overlay = frames_to_show[frame_id].copy()
                overlay[mask] = color
                frames_to_show[frame_id] = cv2.addWeighted(overlay, 0.5, frames_to_show[frame_id], 0.5, 0)

                # add bbox and text to the mask
                coords = cv2.findNonZero(mask.astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    label = f"{category}:{instance_id}"
                    cv2.rectangle(frames_to_show[frame_id], (x, y), (x + w, y + h), color, 2)
                    text_pos = (x, y - 5)
                    if y - 5 < 10:
                        text_pos = (x, y + 15)
                    cv2.putText(frames_to_show[frame_id], label, text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(frames_to_show[frame_id], label, text_pos, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            color_idx += 1

    # save the output video
    with tempfile.TemporaryDirectory() as temp_dir:
        for frame_id, image in enumerate(frames_to_show):
            cv2.imwrite(os.path.join(temp_dir, f'{frame_id + 1}.png'), image)
        os.system(f"ffmpeg -f image2 -i {os.path.join(temp_dir, '%d.png')} {output_path}")
        
