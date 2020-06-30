import os
import cv2
import numpy as np

class RCNN_Mask:
    """ Implementation of semantic mask.
    Each mask has a pillow or numpy image (grayscale), a label (class) and a score (How certain we are that the mask correspond to this class)"""

    def __init__(self, mask, label, score):
        """
        Create a new RCNN mask
        @param mask: Pillow image of mask or numpy array
        @param label: Class for this mask (e.g. 3)
        @param score: Average score for this mask [0,1]
        """
        self.mask = mask
        self.label = label
        self.score = score

    def save(self, number, path, instance_numb):
        """
        Saves the mask as .png in the folder path/number/instance_numb_label_score.png

        @param number: a number that will be the folder name for this mask. Usually frame number is used
        @param path: path where the mask should be stored
        @param instance_numb: instance id for this mask.
        """
        s = path + "/%05d" % number
        if not os.path.exists(s):
            os.mkdir(s)
        name = str(instance_numb) + "_" + str(self.label) + "_" + str(self.score) + ".png"

        if type(self.mask) is not np.ndarray:
            self.mask.save(s + "/" + name)
        else:
            cv2.imwrite(s + "/" + name, self.mask)