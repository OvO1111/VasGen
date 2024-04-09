import os, sys
sys.path.append("/nas/dailinrui/cmuVesselCodev2/vesselDA")

import numpy as np
import v3.utils as utils

from tqdm import tqdm
from typing import Iterable
from collections import defaultdict
from numpy.linalg import norm, svd
from numpy.random import choice, randint, normal
from scipy.ndimage import binary_erosion, distance_transform_edt

from v3.trace_algorithms.astar import Astar
from v3.trace_algorithms.bspline import Bspline

reader = utils.NiftiReader()


class DynamicTracker:
    def __init__(self, 
                 mask: np.ndarray, 
                 start: Iterable, 
                 start_direction: np.ndarray,
                 end_region: np.ndarray, 
                 n_end_points: int, 
                 **tracker_kwargs):
        
        self.mask = mask
        self.start_point = start
        self.start_direction = start_direction
        self.n_end_points = n_end_points
        self.hyper_params = tracker_kwargs
        self.layered_vessels = []
        
        self.end_points = end_region[choice(np.arange(len(end_region)), size=n_end_points, replace=False)]
        
        self.k = tracker_kwargs.get("k", 2.55)
        if tracker_kwargs.__contains__("mask_of_mask"):
            self.bounding_box = utils.bounding_box(tracker_kwargs["mask_of_mask"], outline=5)
            self.mask = self.mask[*self.bounding_box.cropper]
            self.start_point = self.bounding_box.transformer(self.start_point)
            self.end_points = self.bounding_box.transformer(self.end_points)
        
        self.tracker = Astar(self.mask)
        self.meshgrid = np.array(np.meshgrid(*[np.arange(self.mask.shape[x]) for x in range(3)], indexing="ij"))
        
    def _paint(self):
        global reader
        
        label = reader.zeros(dtype=np.uint8)
        mask_edt = distance_transform_edt(self.mask == 0)
        for i, v in enumerate(self.layered_vessels):
            label[*np.array(v["vessel"]).T] = i + 1
            label_edt = distance_transform_edt(label != i + 1)
            label[(mask_edt > 1) & (label_edt < max(5-i, 2))] = i + 1
        reader.save_to_disk(label.astype(np.uint8))
        
    def process(self):
        interpolator = Bspline(self.layered_vessels)
        processed_vessels = interpolator.connect()
        
    def run(self):
        base_r = 4
        tracking_index = 0
        bifur_threshold = 5
        
        vessel_to_track = [dict(r=base_r,
                                end=self.end_points,
                                start=self.start_point,
                                start_dir=self.start_direction)]
        
        while tracking_index < 10:
            r0 = vessel_to_track[tracking_index]["r"]
            ends = vessel_to_track[tracking_index]["end"]
            start = vessel_to_track[tracking_index]["start"]
            start_dir = vessel_to_track[tracking_index]["start_dir"]
            
            u, s, _ = svd(ends.T @ ends / len(ends))
            main_axis = u[np.diag(s).argmax()]
            
            # 1) compute bifurcation zone
            if len(ends) > 1:
                sum_xi = np.zeros(self.mask.shape)
                sum_xi_sq = np.zeros(self.mask.shape)
                for end in tqdm(ends, total=len(ends)):
                    s = self.meshgrid - end[:, None, None, None]
                    s = s / norm(s, axis=0)
                    xi = np.dot(s.transpose(1, 2, 3, 0), start_dir)
                    sum_xi += xi
                    sum_xi_sq += xi ** 2
                std = sum_xi_sq - sum_xi ** 2 / len(ends)
                std[np.isnan(std)] = 0
                bifurcation_zone = std > bifur_threshold * 0.9 ** tracking_index
            else:
                bifurcation_zone = np.zeros(self.mask.shape)
            
            # 2) track using astar from start to bifurcation zone
            path = self.tracker.connect(start,
                                        np.mean(ends, axis=0),
                                        bifurcation_zone, start_dir)
            
            # 3) identify ends and start directions for next iteration
            path_end = path[-1]
            if len(path) == 10: continue
            
            e1 = ends[np.dot(ends - path_end[None], main_axis) > 0]
            e2 = ends[np.dot(ends - path_end[None], main_axis) <= 0]
            q1_to_q2_ratio = len(e1) / len(e2) if len(e2) != 0 else np.inf
            q2_to_q1_ratio = len(e2) / len(e1) if len(e1) != 0 else np.inf
            r1 = (r0 ** self.k / (1 + q2_to_q1_ratio ** (self.k / 2))) ** (1 / self.k)
            r2 = (r0 ** self.k / (1 + q1_to_q2_ratio ** (self.k / 2))) ** (1 / self.k)
            
            # 4) add next vessel param to list
            end_dir = utils.get_direction(path, por_or_index=-1)
            cos_theta = lambda x, y: (r0 ** 4 + x ** 4 - y ** 4) / (2 * x ** 2 * r0 ** 2)
            unit_radius_ball = np.asarray(
                np.meshgrid(*[np.arange(-2, 3) for _ in range(3)], indexing="ij")
            ).reshape(3, -1).T
            
            if r1 > 1.:
                cos = cos_theta(r1, r2)
                orient_to_theta = np.abs(np.dot(unit_radius_ball, end_dir) / norm(unit_radius_ball, axis=1) - cos) < 0.1
                orient_to_main_axis = np.dot(unit_radius_ball[orient_to_theta], main_axis).argmax()
                next_dir = unit_radius_ball[orient_to_theta][orient_to_main_axis] /\
                    norm(unit_radius_ball[orient_to_theta][orient_to_main_axis])
                
                vessel_to_track.append(dict(r=r1,
                                            end=e1,
                                            start=path_end,
                                            start_dir=next_dir))
            if r2 > 1.:
                cos = cos_theta(r2, r1)
                orient_to_theta = np.abs(np.dot(unit_radius_ball, end_dir) / norm(unit_radius_ball, axis=1) - cos) < 0.1
                orient_to_main_axis = np.dot(unit_radius_ball[orient_to_theta], main_axis).argmax()
                next_dir = unit_radius_ball[orient_to_theta][orient_to_main_axis] /\
                    norm(unit_radius_ball[orient_to_theta][orient_to_main_axis])
                    
                vessel_to_track.append(dict(r=r2,
                                            end=e2,
                                            start=path_end,
                                            start_dir=next_dir))
                
            self.layered_vessels.append(dict(r=r0,
                                            layer=0,
                                            vessel=path))
            tracking_index += 1
            self._paint()
            continue
        
        # self.process()
        

if __name__ == "__main__":
    import SimpleITK as sitk
    base_focus_target_num = 100000
    base_non_focus_target_density = 100
    
    reader.load("/nas/dailinrui/dataset/cmu/v2/mask/Dataset000CMU_00001_0000.nii.gz", 
                "/nas/dailinrui/targets.nii.gz")
    mask = reader.array
    
    edt = distance_transform_edt(mask == 108)
    end_region = np.argwhere((edt > 0) & (edt < 2))
    tracker = DynamicTracker(mask, 
                             start=np.asarray([272, 272, 296]),
                             start_direction=np.asarray([0, 0, -1]),
                             end_region=end_region, n_end_points=10)
    tracker.run()