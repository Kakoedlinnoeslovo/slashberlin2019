import yaml
import tempfile
import utils

import matplotlib.pyplot as plt
import os
import traceback

import numpy as np
from PIL import Image
from skimage.segmentation import slic, felzenszwalb
import time
from skimage import io
from tqdm import tqdm

from app.style_transfer.CIHP_master.PGN_single import PGN
from app.style_transfer.CIHP_master.utils_ import preds2coloredseg
from app.style_transfer import config

import cv2

from app.wrappers import pathes_class

REPO_CONFIG = pathes_class.repoConfig()
#DIR = REPO_CONFIG["DIR"]
#IMG_ROOT = REPO_CONFIG["IMG_ROOT"]
#TEMP_IMAGE = REPO_CONFIG["TEMP_IMAGE"]
WEIGHT_PATH = REPO_CONFIG["WEIGHT_PATH"]
#OUT_PATH = REPO_CONFIG["OUT_PATH"]

class SegmentationModel:
    def __init__(self, weights_dir):
        self._model = PGN()
        self._model.build_model(n_class=20, path_model_trained=weights_dir,
                                tta=[0.75, 0.5],
                                img_size=(512, 512), need_edges=False)
        self._threshold = 0.7

    def predict(self, image_path: str) -> (np.ndarray, np.ndarray):
        # image = Image.fromarray(image, 'RGB')

        scores = self._model.predict(image_path)  # W x H x 20

        probs = np.ascontiguousarray(self._softmax(scores, axis=2).transpose(2, 0, 1))

        # Merge Shoes
        probs[-2] += probs[-1]
        probs = probs[:-1]

        assert len(probs) == len(config.SegmentationClassNames.ALL), len(probs)

        classes_ = np.argmax(probs, 0).astype(np.uint8)

        confidences = []
        for i in range(len(config.SegmentationClassNames.ALL)):
            nc = np.sum(classes_ == i)
            conf = 0 if nc == 0 else probs[i][classes_ == i].sum() / nc

            confidences.append(conf)
        confidences = np.array(confidences)

        image = Image.open(image_path)
        labeled_image = np.array(preds2coloredseg(probs, image, out_format='gray'),
                                 dtype=np.uint8)
        return labeled_image, probs

    @staticmethod
    def _softmax(x, axis=0):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis)[..., None])
        return e_x / e_x.sum(axis=axis)[..., None]





def save_image(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
    return path



def grabcut_algo(mask, im):

    scene_mask = np.zeros((im.shape[0], im.shape[1]))

    mask = mask.astype(np.uint8)
    graph_rgb = slic(im, n_segments=1000, compactness=10., max_iter=10, sigma=0)
    rough_mask = np.zeros(im.shape[:2])
    for j in range(len(np.unique(graph_rgb))):
        cluster = np.zeros(mask.shape)
        cluster[graph_rgb == j] = 1
        iou = np.sum(cluster * mask) / np.sum(cluster)
        rough_mask[cluster * mask != 0] = iou



    guidance_mask = np.zeros(im.shape[:2], np.uint8)
    guidance_mask[rough_mask < np.maximum(0.1, np.min(rough_mask))] = 2
    guidance_mask[rough_mask == 0] = 0
    guidance_mask[rough_mask >= np.maximum(0.1, np.min(rough_mask))] = 3
    guidance_mask[rough_mask >= np.minimum(0.6, np.max(rough_mask))] = 1


    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        instance, bgdModel, fgdModel = cv2.grabCut(im, guidance_mask, None, bgdModel, fgdModel, 5,
                                                       cv2.GC_INIT_WITH_MASK)
    except:
        traceback.print_exc()
        return None

    instance = np.where((instance == 2) | (instance == 0), 0, 1).astype('uint8')

    instance = cv2.erode(instance, kernel=np.ones((5, 5), np.uint8), iterations=1)
    scene_mask[instance != 0] = 1
    return scene_mask

_segmentation_model = SegmentationModel(WEIGHT_PATH)

def _get_segmentation_mask(in_path):
    mask, probas = _segmentation_model.predict(in_path)
    return mask, probas

def get_accurate_mask(img_path):
    img = cv2.imread(img_path)[:, :, ::-1]

    segmentation_mask, segmentation_probas = _get_segmentation_mask(img_path)
    t_shirt_mask_1d = segmentation_mask == 5  # 5 is a class of t-shirt

    scene_mask = grabcut_algo(t_shirt_mask_1d, img)
    if scene_mask is None:
        return None
    else:
        scene_mask = scene_mask.astype(np.bool)

        scene_mask = np.expand_dims(scene_mask, 2)
        scene_mask = np.repeat(scene_mask, 3, 2)

        fg = img * scene_mask

        # Convert black pixels to white
        for i in range(fg.shape[0]):
            for j in range(fg.shape[1]):
                if ((fg[i, j] == 0).all() or (fg[i, j] == 0).all() or (fg[i, j] == 0).all()):
                    fg[i, j] = 255  # Just change R channel
        return fg


if __name__ == "__main__":
    folder_list = os.listdir(IMG_ROOT)
    for folder in tqdm(folder_list):
        f_path = os.path.join(IMG_ROOT, folder)
        img_list = os.listdir(f_path)
        out_path = os.path.join(OUT_PATH, folder)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        for img_p in img_list:
            img_path = os.path.join(f_path, img_p)
            mask = get_accurate_mask(img_path)
            if mask is not None:
                img_out_path = os.path.join(out_path, img_p)
                io.imsave(img_out_path, mask)
            else:
                continue
