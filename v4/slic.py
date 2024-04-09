import os, sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd
from v3 import utils
from torch import nn
from tqdm import tqdm
from scipy.ndimage import binary_dilation


class SLIC:
    def __init__(self, nd_image: np.ndarray, alpha: int | float, k: int = 1 << 15):
        """
        SLIC implementation, 
        alpha is the weight coeff of color wrt distance
        s is the distance between adjacent pixel centers
        k is the destinated superpixel number ((k ** 1/3) at each axis)
        """
        self.k = k
        k_ = k ** (1 / 3)
        self.image = nd_image
        self.dist_coeff = alpha
        self.s = (np.ceil(self.image.shape[0] / k_).astype(int), 
                  np.ceil(self.image.shape[1] / k_).astype(int), 
                  np.ceil(self.image.shape[2] / k_).astype(int))
        
        xs = [np.arange(0, self.image.shape[i]) for i in range(3)]
        self.image_to_ind = np.asarray(np.meshgrid(*xs, indexing="ij"))
        self.pixel_assignment = np.zeros(self.image.shape, dtype=np.uint32)
        self.pixel_dist = np.full(self.image.shape, dtype=np.float32, fill_value=1e8)
        
        self._gen_centers()
        self.cluster_ctr = np.vstack([self.cluster_ctr,
                                      self.image[*self.cluster_ctr.reshape(3, -1)].reshape(self.cluster_ctr.shape[1:])[np.newaxis]])
        
    def _gen_centers(self):
        grad = np.linalg.norm(np.asarray(np.gradient(self.image)), axis=0)
        tensorized_grad = torch.from_numpy(grad).to(torch.float32).unsqueeze(0).unsqueeze(0)
        _, pooled_ind = nn.functional.max_pool3d_with_indices(-tensorized_grad, kernel_size=3, stride=self.s)
        self.cluster_ctr = np.stack(np.unravel_index(pooled_ind.squeeze().numpy(), shape=self.image.shape, order="C"))

        for i_center, cluster_center in self._iter:
            x, y, z = cluster_center
            self.pixel_assignment[*cluster_center] = i_center
    
    def repeat(repeat_itr=10):
        def wrapper(func):
            def impl_(self, *args, **kwargs):
                for _ in tqdm(range(repeat_itr), desc="main loop"):
                    ret = func(self, *args, **kwargs)
                
                unassigned_pixels = np.argwhere(self.pixel_assignment == 0)
                print(f"exec final assignment, {(self.pixel_assignment > 0).sum()} assigned {len(unassigned_pixels)} unassigned")
                
                unassigned_pixels_feature = np.hstack([unassigned_pixels, self.image[*unassigned_pixels.T][:, np.newaxis]])
                for i_center, cluster_center in tqdm(self._iter, desc="residual loop", total=np.prod(self.cluster_ctr.shape[1:])):
                    dist_dist = ((unassigned_pixels_feature[:, :-1] - cluster_center[np.newaxis, :-1]) ** 2 /\
                        np.array(self.s)[np.newaxis] ** 2).sum(-1)
                    hu_dist = np.abs(unassigned_pixels_feature[:, -1] - cluster_center[-1]) ** 2
                    total_dist = hu_dist + dist_dist * self.dist_coeff ** 2
                    
                    unassigned_dist = self.pixel_dist[*unassigned_pixels.T]
                    assignment_array = total_dist < unassigned_dist
                    assignment_indices = unassigned_pixels[assignment_array]
                    self.pixel_dist[*assignment_indices.T] = total_dist[assignment_array]
                    self.pixel_assignment[*assignment_indices.T] = i_center
                
                print("SLIC algorithm completed, exiting ...")
                return ret
            return impl_
        return wrapper
    
    @property
    def _iter(self):
        i = 0
        for x in range(self.cluster_ctr.shape[1]):
            for y in range(self.cluster_ctr.shape[2]):
                for z in range(self.cluster_ctr.shape[3]):
                    i += 1
                    yield i, self.cluster_ctr[:, x, y, z]
    
    @repeat()
    def run(self):
        for i_center, cluster_center in self._iter:
            x1 = np.arange(max(round(cluster_center[0] - self.s[0]), 0),
                           min(round(cluster_center[0] + self.s[0]), self.image.shape[0] - 1)).astype(int)
            x2 = np.arange(max(round(cluster_center[1] - self.s[1]), 0),
                           min(round(cluster_center[1] + self.s[1]), self.image.shape[1] - 1)).astype(int)
            x3 = np.arange(max(round(cluster_center[2] - self.s[2]), 0),
                           min(round(cluster_center[2] + self.s[2]), self.image.shape[2] - 1)).astype(int)
            
            cluster_neighbor = np.asarray(list(map(np.ravel, np.meshgrid(x1, x2, x3, indexing="ij")))).T
            
            cluster_neighbor_hus = self.image[*cluster_neighbor.T][:, np.newaxis]
            cluster_neighbor_features = np.hstack([cluster_neighbor, cluster_neighbor_hus])
            
            dist_dist = ((cluster_neighbor_features[:, :-1] - cluster_center[np.newaxis, :-1]) ** 2 /\
                np.array(self.s)[np.newaxis] ** 2).sum(-1)
            hu_dist = np.abs(cluster_neighbor_features[:, -1] - cluster_center[-1]) ** 2
            total_dist = hu_dist + dist_dist * self.dist_coeff ** 2
            
            cluster_neighbor_dist = self.pixel_dist[*cluster_neighbor.T]
            assignment_array = total_dist < cluster_neighbor_dist
            assignment_indices = cluster_neighbor[assignment_array]
            self.pixel_dist[*assignment_indices.T] = total_dist[assignment_array]
            self.pixel_assignment[*assignment_indices.T] = i_center
            
            if len(assignment_array) > 0:
                cluster_center[:] = np.concatenate([assignment_indices.mean(0),
                                                    cluster_neighbor_hus[assignment_array].mean()[None]])
        
        return True
            
            
if __name__ == "__main__":
    import SimpleITK as sitk
    image = sitk.GetArrayFromImage(sitk.ReadImage("/nas/dailinrui/dataset/nnUNetv2/nnUNet_raw/Refset10CMU/imagesTs/Refset10CMU_0001_0000.nii.gz"))
    label = sitk.GetArrayFromImage(sitk.ReadImage("/nas/dailinrui/dataset/nnUNetv2/nnUNet_raw/Refset10CMU/labelsTs/Refset10CMU_0001.nii.gz"))
    image = utils.window_norm(image)
    bbox = utils.bounding_box(label > 0, outline=10)
    
    image = image[*bbox.cropper]
    slic = SLIC(image, alpha=1e-1)
    slic.run()
    superpxl = slic.pixel_assignment
    # superpxl = bbox.inv_transformer(superpxl)

    # sitk.WriteImage(sitk.GetImageFromArray(image), "/nas/dailinrui/ct.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(superpxl), "/nas/dailinrui/label_.nii.gz")