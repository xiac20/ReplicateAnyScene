# import numpy as np
# from geometry_utils import compute_surface_area_from_pointmap

# class FrameMask:
#     """
#     Represents a binary mask associated with a specific video frame.
    
#     Attributes:
#         frame_id (int): The unique identifier or index of the video frame.
#         mask (np.ndarray): A 2D binary numpy array (H, W) representing the mask.
#         surface_area (float): 3D surface area of the mask.
#     """
    
#     def __init__(self, frame_id: int, mask: np.ndarray, pointmap: np.ndarray = None):
#         self.frame_id = frame_id
#         self.mask = mask
#         if pointmap is None:
#             self.surface_area = -1
#         else:
#             self.update_surface_area(pointmap)
    
#     def update_surface_area(self, pointmap: np.ndarray = None):
#         self.surface_area = compute_surface_area_from_pointmap(pointmap, self.mask)
            
#     @property
#     def pixel_area(self):
#         return int(np.sum(self.mask))

#     def __repr__(self):
#         return f"FrameMask(frame_id={self.frame_id}, surface_area={self.surface_area})"

# class instance:
#     """
#     Represents a binary mask associated with a specific video frame.
    
#     Attributes:
#         masks (list): A list of FrameMask object
#         real_points (np.ndarray): 
#     """