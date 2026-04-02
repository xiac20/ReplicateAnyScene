from vggt.utils.load_fn import load_and_preprocess_images
import os
import re
import numpy as np

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
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            os.system(f"ffmpeg -i {video_path} -vsync 0 {temp_dir}/frame_%04d.png")
            return load_video_frames(temp_dir, max_frames)