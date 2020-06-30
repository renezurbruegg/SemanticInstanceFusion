import numpy as np
import scipy
import torch
from PIL import Image

import src.MaskRCNNImpl.Model as Model


####################################################################################
#      Contains program code to extract Semantic Masks for an RGB image            #
####################################################################################
from src.SemanticMasks.RCNN_mask import RCNN_Mask

class Segmenter:
    """
    Class to extract RCNN Masks from RGB images.
    """

    def __init__(self, path_to_data, model_path, img_size, device, label_path="labels_40.csv", batch_size=1,
                 mask_threshold=0.5, score_threshold=0.5):
        """
        Creates a new segmenter object used to extract masks from rgb images

        @param path_to_data: Path to the folder that contasin the RGB images
        @param model_path: Path to the pretrained backbone model
        @param img_size: Image size as tuple. e.g. (640,480)
        @param device: Preferred pytorch device for the segmentation (preferably GPU)
        @param label_path: Path where labels,class_name,color are stored
        @param batch_size: How many images to segment in one batch
        @param mask_threshold: Points inside a mask with a lower threshold will be removed
        @param score_threshold: Masks with lower average score than this threshold will be discarded
        """
        self.model = None
        self.device = device
        self.load_model(model_path)
        self.path_to_data = path_to_data
        self.img_size = img_size
        self.label_path = label_path
        self.batch_size = batch_size
        self.classes = Model.get_classes(label_path)
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold

    def load_model(self, model_path):
        """
        Loads the backbone model
        @param model_path: path to the backbone model
        """
        self.model = Model.load_weights(path=model_path, device=self.device)
        self.model.eval()

    def get_mask_for_images(self, img_list):
        """
        Returns a list of mask for the given images
        @param img_list: list of image numbers to be segmented
        @return: List of List of Mask RCNN Masks. [ [RCNN_Mask, RCNN_Mask],[...],...]
        """
        mask_list = []
        with torch.no_grad():
            for i in range(0, img_list.__len__(), self.batch_size):
                imgs = np.array([self.load_img(img_list[i + number]) for number in range(self.batch_size)])
                imgs = torch.tensor(imgs).to(self.device)
                results = self.model.forward(imgs)
                for j in range(results.__len__()):
                    mask_list.append(self.get_mask_from_result(results[j]))

        return mask_list

    def get_mask_from_result(self, result, erode=False):
        """
        Returns a list of RCNN masks for a result tensor from the mask rcnn
        @param result: output of mask rcnn
        @param erode: whether or not to use erosion
        @return: list of RCNN Masks
        """
        masks = result['masks'].cpu().detach().numpy()
        labels = result['labels'].cpu().detach().numpy()
        scores = result['scores'].cpu().detach().numpy()

        rcnn_masks = []

        # All pixels in the mask with a value bigger than this threshold are
        # assigned to the binary mask.

        # All detections with a score bigger than this threshold are used as actual
        # detections.

        # Extract predicted boxes and masks.

        masks[masks < self.mask_threshold] = 0
        for i in range(masks.shape[0]):
            if scores[i] > self.score_threshold:
                # Create non-binary object mask and draw it into the test image.
                mask = (masks[i, 0, :, :] * 255).astype(np.uint8)
                if erode:
                    mask = scipy.ndimage.morphology.grey_erosion(mask, size=(3, 3))
                mask = Image.fromarray(mask, mode='L')
                rcnn_masks.append(RCNN_Mask(mask, labels[i], scores[i]))

        return rcnn_masks

    def load_img(self, number):
        """
        Loads an image for a given number
        @param number: the number of the image
        @return: the image as numpy array
        """
        path = self.path_to_data + "/%05d.jpg" % number
        # Prepare input image for running through the model.
        img = np.array(Image.open(path).convert('RGB')).astype('float32') / 255.
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        return img
