import os
import re

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.SemanticMasks.RCNN_mask import RCNN_Mask

try:
    from src.SemanticMasks import segmenter
except ModuleNotFoundError as e:
    print("Could not import Module \n SemanticInstanceFusion Will not be able to segment masks", e)

with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

pattern = re.compile("\d+_\d+_.+")


#############################################################################################
#        Provides different ways to obtain semantic masks for a RGB image                   #
#############################################################################################

class MaskFromGroundtruthGenerator:
    def __init__(self, gt_instance_path, gt_label_path, img_size):
        """
        Mask Generator which reads precomputed mask. Does not need a lot of GPU power.
        Dose require the results to be at the given mask_path in the right format

        @param mask_path: path to the mask data
        """
        self.gt_instance_path = gt_instance_path
        self.gt_label_path = gt_label_path
        self.img_size = img_size

        self.label_mapping_to_nyu = self.load_label_mapping_to_nyu()

    def load_label_mapping_to_nyu(self):
        mapping_df = pd.read_csv('evaluation/scannetv2-labels.combined.tsv', sep='\t')
        labels = mapping_df['id'].values
        nyu40ids = mapping_df['nyu40id'].values
        return dict(zip(labels, nyu40ids))

    def getMaskForNumber(self, img_number):
        """
        @param img_number: number of the current image
        @return: masks: R-CNN-Masks for the current image
        """
        instance_path = self.gt_instance_path + "/%d" % img_number + ".png"
        label_path = self.gt_label_path + "/%d" % img_number + ".png"

        instance_image = np.array(Image.open(instance_path).resize(self.img_size, Image.NEAREST)).astype(int)
        label_image = np.array(Image.open(label_path).resize(self.img_size, Image.NEAREST)).astype(int)

        instances = np.unique(instance_image)
        labels = np.zeros(len(instances), dtype=int)
        scores = np.ones(len(instances))

        instance_masks = np.zeros((len(instances),
            instance_image.shape[0], instance_image.shape[1]), dtype=bool)
        for i,inst in reversed(list(enumerate(instances))):
            instance_masks[i,:,:] = (instance_image == inst)
            l = np.unique(label_image[instance_masks[i,:,:]])
            if len(l) > 1:
                print('Warning: Ground truth instance contains multiple labels!')
            labels[i] = int(l[0])

            if labels[i] == 0:
                instances = np.delete(instances, i)
                instance_masks = np.delete(instance_masks, i, axis=0)
                labels = np.delete(labels, i)
                scores = np.delete(scores, i)
            else:
                labels[i] = self.label_mapping_to_nyu[labels[i]]


        instance_masks = 255 * instance_masks
        m = [RCNN_Mask(instance_masks[i,:,:], labels[i], scores[i]) for i in range(len(instances))]
        c = 0

        if params["debug"]["groundtruth_mask"]["save"]:
            for m1 in m:
                c +=1
                m1.save(img_number,params["debug"]["groundtruth_mask"]["path"],c)

        return m


class MaskFromFileGenerator:
    def __init__(self, mask_path):
        """
        Mask Generator which reads precomputed mask. Does not need a lot of GPU power.
        Dose require the results to be at the given mask_path in the right format

        @param mask_path: path to the mask data
        """
        self.mask_path = mask_path

    def getMaskForNumber(self, img_number):
        """
        @param img_number: number of the current image
        @return: masks: R-CNN-Masks for the current image
        """
        path = self.mask_path + "/%05d" % img_number
        masks, labels, scores = load_result(path)
        return [RCNN_Mask(masks[i], labels[i], scores[i]) for i in range(len(masks))]

class DynamicMaskGenerator:
    def __init__(self, rgb_path, model_path, img_size, device, labels_path,
                 mask_path= "NOT A REAL PATH", Save_Flag = False, prefetch_size = 1, batch_size = 1, score_threshold = 0.5, mask_threshold = 0.5, skip = 10):
        """
        Uses a trained Mask-R-CNN to obtain the segmentation masks.

        @param rgb_path: path to the rgb images
        @param model_path: path to the model
        @param img_size: size of the images (will resize images if needed)
        @param device: preferred pytorch device for the segmentation (preferably GPU)
        @param labels_path: path to the csv file with all the labels
        @param mask_path: path to where the mask should be stored if Save_Flag is True
        @param Save_Flag: determines if the computed masks should be saved to the mask_path directory
        @param prefetch_size: determines how many images should be processed at once
        @param batch_size: determines how big the batches are which are given directly to the model
        """

        self.segmenter = segmenter.Segmenter(rgb_path, model_path, img_size, device,
                                             label_path = labels_path, batch_size=batch_size, score_threshold = score_threshold, mask_threshold= mask_threshold)
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.prefetched_masks = {}
        self.rgb_path = rgb_path
        self.mask_path = mask_path
        self.save_flag = Save_Flag
        self.skip = skip

    def getMaskForNumber(self, img_number):
        """
        @param img_number: number of the current image
        @return: masks: R-CNN-Masks for the current image
        """
        print("Dynamic mask generator. Getting mask for image", img_number)
        if img_number in self.prefetched_masks:
            print("found cached mask")
            return self.prefetched_masks.pop(img_number)
        print("no cached mask found. going to prefetch from segmenter")
        img_list = list(range(img_number, img_number+self.prefetch_size*self.skip, self.skip))
        masks = self.segmenter.get_mask_for_images(img_list)
        for j in range(len(masks)):
            mask_list = masks[j]
            mask_list.reverse()
            self.prefetched_masks[img_list[j]] = mask_list
            if self.save_flag:
                s = self.mask_path + "/%05d" % (img_list[j])
                # Creates the directory to store the masks if it does not exist yet.
                if not os.path.exists(s):
                    os.mkdir(s)
                    print("making: " + s)
                count = 0
                for filename in os.listdir(s):
                    file_path = os.path.join(s, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        print("couldn't remove file: ", file_path)
                for m in masks[j]:
                    m.save(img_number+j, self.mask_path, count)
                    count += 1



        return self.prefetched_masks.pop(img_number)


class CachedMaskGenerator:
    def __init__(self, rgb_path, model_path, mask_path, img_size, device, labels_path, prefetch_size = 1,
                 batch_size = 1,  score_threshold = 0.5, mask_threshold= 0.5, skip = 10):
        """
        Mask Generator which will get precomputed masks if they exist and computes them if they don't.

        @param rgb_path: path to the rgb images
        @param model_path: path to the model
        @param mask_path: path to where the mask should be stored if Save_Flag is True
        @param img_size: size of the images (will resize images if needed)
        @param device: preferred pytorch device for the segmentation (preferably GPU)
        @param labels_path: path to the csv file with all the labels
        @param prefetch_size: determines how many images should be processed at once
        @param batch_size: determines how big the batches are which are given directly to the model
        @param mask_multiplier: used for missmatches between input data and masks
        """
        self.dynamic_mask_gen = DynamicMaskGenerator(rgb_path,
                                                     model_path,
                                                     img_size,
                                                     device,
                                                     mask_path = mask_path,
                                                     Save_Flag = True, labels_path = labels_path,
                                                     prefetch_size = prefetch_size,
                                                     batch_size=batch_size,
                                                     score_threshold = score_threshold,
                                                     mask_threshold= mask_threshold,
                                                     skip= skip)
        self.file_mask_loader = MaskFromFileGenerator(mask_path)
        self.mask_path = mask_path


    def getMaskForNumber(self, img_number):
        """
        @param img_number: number of the current image
        @return: masks: R-CNN-Masks for the current image
        """
        try:
            return self.file_mask_loader.getMaskForNumber(img_number)
        except:
            print("Mask not found. Going to segment and store")
            masks = self.dynamic_mask_gen.getMaskForNumber(img_number)
            return masks



def load_result(path):
    """ Loads masks, labels and scores from the given path"""
    masks = []
    labels = []
    scores = []
    for name in os.listdir(path):
        if pattern.match(name):
            mask = Image.open(path + "/" + name)
            n = name.split("_")

            if len(n) < 3:
                continue

            label = int(n[1])
            score = float(n[2][:-4])
            masks.append(mask)
            labels.append(label)
            scores.append(score)

    # order everything be score. lowest score first
    index_scores = np.array([np.arange(len(masks)), scores]).T
    index_scores = index_scores[index_scores[:, 1].argsort()]

    masks_sorted = []
    labels_sorted = []
    scores_sorted = []

    for ind in index_scores[:, 0]:
        ind = int(ind)

        masks_sorted.append(masks[ind])
        labels_sorted.append(labels[ind])
        scores_sorted.append(scores[ind])
    return masks_sorted, labels_sorted, scores_sorted
