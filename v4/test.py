import os, sys
sys.path.append(os.path.join(os.environ.get("oss"), "code", "vessel.da"))

from os.path import *
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import v3.utils as utils
from functools import reduce
from skimage import morphology
from scipy import ndimage

cmu_blank = join(os.environ.get("oss"), "data", "cmu", "v2", "blank")
cmu_mask = join(os.environ.get("oss"), "data", "cmu", "v2", "mask")
nnunet_label = join(os.environ.get("nnUNet_raw"), "Dataset000CMU", "labelsTr")
nnunet_image = join(os.environ.get("nnUNet_raw"), "Dataset000CMU", "imagesTr")

def euclidean_distance(seed, dis_threshold, spacing=[1, 1, 1]):
    threshold = dis_threshold
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed==0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1-euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis

def salt_and_pepper(salt_size, pepper_size, typename="artery"):
    if typename == "artery": foreground_hu_anchor = artery_hu_anchor
    elif typename == "vein": foreground_hu_anchor = vein_hu_anchor
    
    salt = render_vessel_fn((foreground_hu_anchor[0] + foreground_hu_anchor[1] - 10, 20), salt_size)
    pepper = render_vessel_fn((foreground_hu_anchor[0] - foreground_hu_anchor[1] - 10, 20), pepper_size)
    noise = np.concatenate([salt, pepper])
    np.random.shuffle(noise)
    return noise

def gaussian(size, typename="artery"):
    if typename == "artery": foreground_hu_anchor = artery_hu_anchor
    elif typename == "vein": foreground_hu_anchor = vein_hu_anchor
    return np.random.normal(*foreground_hu_anchor, size)

a = [[], []]
v = [[], []]

"""try:
    for label_name in sorted(utils.listdir(nnunet_label)):
        label_reader = utils.NiftiReader(label_name)
        mask_reader = utils.NiftiReader(join(cmu_mask, label_name.split('/')[-1].replace(".nii.gz", "_0000.nii.gz")))
        blank_reader = utils.NiftiReader(join(cmu_blank, label_name.split('/')[-1].replace(".nii.gz", "_0000.nii.gz")))
        
        raw_foreground = label_reader.array
        foreground = raw_foreground.copy()
        background = blank_reader.array
        
        mask = mask_reader.array

        artery_hu_anchor = [background[mask == 7].mean() * 0.8,
                                    background[mask == 7].std()]
        vein_hu_anchor = [background[mask == 8].mean() * 0.8,
                            background[mask == 8].std()]
        a[0].append(artery_hu_anchor[0])
        a[1].append(artery_hu_anchor[1])
        v[0].append(vein_hu_anchor[0])
        v[1].append(vein_hu_anchor[1])
        
        print(artery_hu_anchor, vein_hu_anchor)
except RuntimeError as e:
    pass
finally:
    print(np.mean(a, axis=1), np.mean(v, axis=1))"""

for label_name in tqdm(sorted(utils.listdir(nnunet_label))):
    label_reader = utils.NiftiReader(label_name)
    mask_reader = utils.NiftiReader(join(cmu_mask, label_name.split('/')[-1].replace(".nii.gz", "_0000.nii.gz")))
    blank_reader = utils.NiftiReader(join(cmu_blank, label_name.split('/')[-1].replace(".nii.gz", "_0000.nii.gz")))
    
    raw_foreground = label_reader.array
    foreground = raw_foreground.copy()
    background = blank_reader.array
    
    mask = mask_reader.array

    artery_hu_anchor = np.asarray([118.033, 65.213]) * 0.8
    vein_hu_anchor = np.asarray([92.814, 50.209]) * 0.8
    abdomen_background_hu_anchor = [60, 60]
    render_vessel_fn = lambda param, size: (np.random.random(size) - 0.5) * param[1] + param[0]

    bounding_box = utils.bounding_box(foreground, outline=5)
    cp_foreground = foreground[*bounding_box.cropper]
    combine = background[*bounding_box.cropper].copy()
    foreground_size = (foreground > 0).sum()
    combine[cp_foreground > 0] = salt_and_pepper(salt_size=(round(foreground_size * 0.7),),
                                                    pepper_size=(foreground_size-round(foreground_size * 0.7)))
    combine = ndimage.zoom(combine, 2., order=0)
    cp_foreground_ = ndimage.zoom(cp_foreground, 2, order=0)
    smooth = ndimage.gaussian_filter(combine, sigma=1.2, radius=3)
    combine[cp_foreground_ > 0] = smooth[cp_foreground_ > 0]
    combine = ndimage.zoom(combine, .5, order=3)

    cp_foreground_expand = cp_foreground > 0
    background[*bounding_box.cropper][cp_foreground_expand] = combine[cp_foreground_expand]

    dist = ndimage.distance_transform_edt(foreground > 0, sampling=blank_reader.spacing)
    surface = (dist > 0) & (dist < .8)
    background[surface] = background[surface] * 0.7

    foreground[*bounding_box.cropper][:] = cp_foreground
    erosion_vessel_mask_array = ndimage.binary_erosion(foreground, structure=np.ones((2,2,2)))
    dis = euclidean_distance(erosion_vessel_mask_array, 3, spacing=label_reader.spacing)
    background = background * dis + blank_reader.array * (1 - dis)
    label_reader.save_to_disk(background, join(nnunet_image, label_name.split('/')[-1].replace(".nii.gz", "_0000.nii.gz")))

    new_label = np.zeros(foreground.shape, dtype=np.uint8)
    for i in range(1, 7):
        new_label[(raw_foreground == i) & (foreground > 0)] = i
    label_reader.save_to_disk(new_label, label_name)