
import os
import time
from collections import Counter

import numpy as np
import yaml
from PIL import Image

from src.SemanticMasks.instance_generator import InstanceGenerator

with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)



#############################################################################################
#     Returns a class image that can be used in the fusion process  for an image number     #
#############################################################################################

class Preprocessor:
    def __init__(self, rgb_path, mask_generator, depth_path, pose_path, superpixel_segmentor, score_threshold = 0.7,
                 allowed_classes=[-1], use_depth = False, depth_f = 100, use_sp = True, num_classes = 40, instance_threshold = 0.7, rcnn_threshold = 40, rcnn_threshold_2 = 50):
        """
        Creates a new preprocessor object
        @param rgb_path: Path to the color images
        @param mask_generator: MaskGenerator that returns RCNN_Masks for a image number
        @param depth_path: Path to the depth image
        @param pose_path: Path to the pose
        @param superpixel_segmentor: class that generates superpixel (can also be None if no superpixels are used)
        @param score_threshold: masks with a value lower than this score will be discarded
        @param allowed_classes: list with classes that are allowed to be fused ([-1] if all classes should be fused)
        @param use_depth: flag if the depth image should be use to improve the mask
        @param depth_f:
        @param use_sp: flag if superpixels should be used to improve generatoed masks
        @param num_classes: how many classes there are in total
        @param instance_threshold: how many instances are avaiable per class
        @param rcnn_threshold:
        @param rcnn_threshold_2:
        """
        self.instance_generator = InstanceGenerator(intrinsics_path = params['camera_intrinsics_path'])
        self.allowed_classes = allowed_classes
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.pose_path = pose_path
        self.score_threshold = score_threshold
        self.mask_generator = mask_generator
        self.fuse_instances = False
        self.use_depth = use_depth
        self.use_sp = use_sp
        self.superpixel_segmentor = superpixel_segmentor
        self.depth_f = float(depth_f)
        self.num_classes = num_classes + 1
        self.instance_threshold = instance_threshold
        self.rcnn_threshold = rcnn_threshold
        self.rcnn_threshold_2 = rcnn_threshold_2



    def get_used_classes(self):
        """
        @return: Classes which have been found
        """
        return self.instance_generator.get_used_classes()

    def combine_instances(self, _class, orig, rest):
        """
        Replaces all instances with id contained in rest with orig
        @param _class: class whose instances we want to combine
        @param orig: original id
        @param rest: ids to replace as list
        """
        self.instance_generator.combine_instances(_class, orig, rest)
        self.fuse_instances = False

    def get_superpixel_score(self, superpixels, mask):

        # Increase assignment by one to get rid of index zero as zero is reserved for
        # the area outside the mask in assignment_masked.
        superpixels += 1
        superpixels_masked = superpixels * mask

        # Array holding the scores
        score = np.zeros(superpixels.shape)

        # The Counter objects allow to count the number of pixels belonging to each
        # superpixel in the initial assignment as well as in the assignment only inside
        # the mask.
        occurence = Counter(superpixels.flatten())
        occurence_masked = Counter(superpixels_masked.flatten())

        # Iterate through every superpixel in the mask and assign the score to each
        # superpixel. The score for superpixel i is the ratio of pixels assigned to
        # superpixel i inside the mask to the number of pixels assigned to superpixel i.
        for i in np.unique(superpixels_masked)[1:]:
            count = occurence.get(i)
            count_masked = occurence_masked.get(i)
            score[superpixels == i] = count_masked / count

        return score

    def get_superpixels(self, depth):
        slic_input = np.concatenate([depth[:,:, None], depth[:,:, None], depth[:,:, None]], axis=2)
        slic_input = (slic_input / np.max(slic_input) * 255).astype(np.uint8)
        return self.superpixel_segmentor.iterate(slic_input)

    def get_images(self, img_number, use_masks=True):
        """
        @param img_number: Number of the current image
        @param use_masks: flag if RCNN masks should be used. If not, only RGB reconstruction will be done
        @return: color_image as a numpy array, the image with class information, depth image, pose of the camera
        """
        # open depth image

        depth_im = np.array(Image.open(self.depth_path + "/%05d.png" % img_number)).astype(float)

        try:
            color_image = Image.open(self.rgb_path + "/%05d.jpg" % img_number)
        except:
            color_image = Image.open(self.rgb_path + "/%05d.png" % img_number)

        pose = np.loadtxt(self.pose_path + "/%05d.txt" % img_number)

        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0

        if not use_masks:
            return np.array(color_image), np.array(color_image)*0, depth_im, pose

        if self.use_sp:
            superpixels = self.get_superpixels(depth_im)
            class_img = self.create_class_image(pose, depth_im, color_image.size, img_number, superpixels)
        else:
            class_img = self.create_class_image(pose, depth_im, color_image.size, img_number, None)

        return np.array(color_image), class_img, depth_im, pose

    def create_class_image(self, pose, depth_im, size, img_number, superpixels):
        """ Creates a class image: Just a regular RGB image, containing the class number as G, and the Score as B value
        @param pose: the pose of the camera (numpy array)
        @param depth_im: the depth image (numpy array)
        @param size: size of the images
        @param img_number: number of image that we are working with (for debugging)
        @param superpixels: flag if superpixels should be used
        @return: class image as numpy 3D array
        """
        masks = self.mask_generator.getMaskForNumber(img_number)
        # Create a new image to draw on
        box_image = Image.new('RGB', size, (0, 0, 0))

        # Convert to numpy array, since using Pillow created artifacts while filling bitmap
        img = np.asarray(box_image).astype(int)

        still_hot = []
        t1 = time.time()
        for rcnn_mask in masks:
            label = rcnn_mask.label
            score = rcnn_mask.score
            non_bin_mask = np.array(rcnn_mask.mask)

            if (score < self.score_threshold) or ((int(label) not in self.allowed_classes) and -1 not in self.allowed_classes):
                continue
            if self.use_sp:
                score = score * self.get_superpixel_score(superpixels, non_bin_mask >= self.rcnn_threshold)
            if self.use_depth:
                # altering mask with depth image
                temp = np.where(non_bin_mask >= self.rcnn_threshold, depth_im, np.nan)
                median = np.nanmedian(temp)
                std_dev = np.nanstd(temp)
                d_score = depth_im - median
                m = (non_bin_mask / (self.depth_f /(std_dev*std_dev+0.1) * d_score*d_score + 1.)*score).astype("uint8")
            else:
                m = non_bin_mask * score


            mask = np.logical_and(m >= img[:,:,2], m >= self.rcnn_threshold_2)

            # Leaves out this mask if the mask is empty.
            if (np.sum(mask) == 0):
                continue

            #
            inst_id, _ = self.instance_generator.get_id_for_class_mask(label, mask, depth_im, pose,still_hot, threshold=self.instance_threshold, factor = 0.1)
            still_hot.append([label,inst_id])
            # Add instance to newly created instances
            #print(label, " / ", inst_id)


            # Set R channel to instance
            img[mask, 0] = inst_id
            # Set G channel of this mask to the value of the class
            img[mask, 1] = label
            # Set B channel of this mask to the score of the prediction
            img[mask,2] = m[mask]
            #del mask, non_bin_mask, score, label, rcnn_mask
        del masks
        print("postprocessing of masks took", (time.time() - t1) * 1000)
        return img

    def downscale_images(self, img_number):
        """
        Resizes the rgb image to the size of the depth image, in case they do not match
        @param img_number: Number of the current image
        @return: None, overwrites old image
        """
        # open depth image
        depth_im = np.array(Image.open(self.depth_path + "/%05d.png" % img_number)).astype(float)

        load_png = False
        try:
            color_image = Image.open(self.rgb_path + "/%05d.jpg" % img_number)
        except:
            load_png = True
        if load_png:
            color_image = Image.open(self.rgb_path + "/%05d.png" % img_number)

        if depth_im.shape[1] != color_image.size[0] or depth_im.shape[0] != color_image.size[1]:
            print(
                "Images not same size. Going to reshape images. Will overwrite all images to not downsample them again")
            print("from ->(", color_image.size[0], color_image.size[1], ") -> (", depth_im.shape[1], depth_im.shape[0],
                  ")")

            color_image = color_image.resize((depth_im.shape[1], depth_im.shape[0]))
            color_image.save(self.rgb_path + "/%05d.jpg" % img_number)

            print("downscaling masks")
            path = "notapath"
            try: path = self.mask_generator.mask_path + "/%05d" % img_number
            except: print("well that did not work... there is no self.mask_path so I used the path in the mask generator, but seems like this mask generator does not have a mask path... probably should find a better solution")
            for mask in os.listdir(path):
                if ".png" in mask or ".jpg" in mask:
                    img = Image.open(path + "/" + mask)
                    if depth_im.shape[1] != img.size[0] or depth_im.shape[0] != img.size[1]:
                        img.resize((depth_im.shape[1], depth_im.shape[0]), resample=Image.NEAREST).save(path + "/" + mask)
                        print("downscaling mask:", mask)

    def critical_instance_count(self):
        """
        @return: true if instance count is low in instance generator, false otherwise
        """
        return self.instance_generator.critical_instance_count
