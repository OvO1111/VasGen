import os, sys
sys.path.append(__file__)
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import math
import numpy as np
from typing import Any

import v4.utils as utils
from v4.macros import LABEL_MAPPING
from v4.generators.foreground import ForegroundGenerator

from scipy import ndimage


def euclidean_distance(seed, dis_threshold, spacing=[1, 1, 1]):
    threshold = dis_threshold
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed==0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1-euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis


class BackgroundGenerator:
    def __init__(self, feed):
        if isinstance(feed, ForegroundGenerator):
            self.preliminary_background = feed.background_prototype
            self.foreground = feed.foreground
        elif isinstance(feed, dict):
            self.preliminary_background = feed.get('background')
            self.foreground = feed.get("foreground")
        self.background = self.preliminary_background
        
        self.process()
        
    def process(self):
        # 1. simulate the fade of hu from vessel lumen to vessel wall
        # 2. diffusion
        pass
        
        
if __name__ == "__main__":
    import pathlib as pb
    from tqdm import tqdm
    base = pb.Path(os.path.join(os.environ.get("nnUNet_raw"), "Dataset004CMU"))
    os.makedirs(base / "pimagesTr", exist_ok=True)
    for i in tqdm(range(1, 11)):
        image_file = base / "background" / f"Dataset004CMU_{i:04d}_0000.nii.gz"
        mask_file = base / "labelsTr" / f"Dataset004CMU_{i:04d}.nii.gz"
        totalseg_file = base / "totalseg" / f"Dataset004CMU_{i:04d}.nii.gz"
        generator = BackgroundGenerator(dict(mask=totalseg_file,
                                            foreground=mask_file,
                                            blank=image_file))
        generator.reader.save_to_disk(generator.background, base / "pimagesTr" / f"Dataset004CMU_{i:04d}_0000.nii.gz")