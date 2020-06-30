import time

import cv2
import fast_slic
import networkx as nx
import numpy as np
import yaml
from PIL import Image

from src import Preprocessor, SemInstVolume
from src.VolumeExtractor import VolumeExtractor

with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)


#############################################################################################
#                       Fuses frames into a Semantic instance Volume                       #
#############################################################################################
class SemanticFuser():
    def __init__(self, classes, mask_generator, n_imgs=10, start=1651,
                 path_to_rgb="data",
                 path_to_d="data",
                 path_to_pose="data",
                 score_threshold=0.7,
                 voxel_size=0.03,
                 filter_interval=-1,
                 use_depth = False,
                 depth_f = 100,
                 use_sp = True,
                 fast_slic_comp=1500,
                 instance_threshold = 0.7,
                 rcnn_threshold = 50,
                 rcnn_threshold_2 = 40
                 ):
        """
        Creates a new fuser object
        @param classes: mapping class number to class name
        @param mask_generator: mask generator that should be used to create Semantic Masks
        @param n_imgs: how many images to fuse
        @param start: which image number to start with
        @param path_to_rgb: path to the rgb images
        @param path_to_d: path to the depth images
        @param path_to_pose: path to the pose images
        @param score_threshold: masks with a value lower than this score will be discarded
        @param voxel_size: the voxel size
        @param filter_interval:
        @param use_depth: flag if the depth image should be use to improve the mask
        @param depth_f: flag to specify if depth values should be used to improve masks
        @param use_sp: flag if superpixels should be used to improve generatoed masks
        @param fast_slic_comp:
        @param instance_threshold: how many instances are available per class
        @param rcnn_threshold:
        @param rcnn_threshold_2:
        """
        self.n_imgs = n_imgs
        self.start = start
        self.path_to_rgb = path_to_rgb
        self.path_to_d = path_to_d
        self.path_to_pose = path_to_pose
        self.classes = classes

        self.preprocessor = Preprocessor.Preprocessor(path_to_rgb, mask_generator, path_to_d, path_to_pose,
                                                      fast_slic.Slic(num_components=fast_slic_comp), score_threshold,
                                                      use_depth = use_depth, depth_f=depth_f, use_sp = use_sp, instance_threshold=instance_threshold,
                                                      rcnn_threshold=rcnn_threshold, rcnn_threshold_2= rcnn_threshold_2)

        print("Estimating voxel volume bounds...")
        self.cam_intr = np.loadtxt(params['camera_intrinsics_path'], delimiter=' ')
        vol_bnds = np.zeros((3, 2))

        for i in range(start, start + n_imgs, 10):
            # Read depth image and camera pose
            try:
                depth_im = np.array(Image.open(self.path_to_d + "/%05d.png" % (i))).astype(float)
            except FileNotFoundError as e:
                print('Warning: depth image ', i, ' missing. It was skipped',e)
                continue
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters

            # print( np.max(depth_im))
            depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
            try:
                cam_pose = np.loadtxt(self.path_to_pose + "/%05d.txt" % (i))  # 4x4 rigid transformation matrix
            except IOError as e:
                print('Warning: camera pose ', i, ' is missing. It was skipped',e)
                continue

            # Skips this image if camera pose contains nan or inf values.
            if np.isnan(cam_pose).any() or np.isinf(cam_pose).any():
                print('Warning: camera pose ', i, ' is invalid. It was skipped')
                continue

            # Compute camera view frustum and extend convex hull
            view_frust_pts = SemInstVolume.get_view_frustum(depth_im, self.cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

            # Downscale images if needed
            try:
                self.preprocessor.downscale_images(i)
            except (IOError, FileNotFoundError):
                print('Warning: file ', i, ' missing. It was skipped')
                continue

        print("Volume Bounds: " + str(vol_bnds))
        print("Initializing voxel volume...")
        self.tsdf_vol = SemInstVolume.TSDFVolume(vol_bnds, voxel_size=voxel_size, filter_interval=filter_interval,
                                                 use_gpu=True)

    def fuse_images(self, i, use_masks = True):
        """
        Fuses a image with image number i
        @param i: image number
        @param use_masks: Flag if masks for this images should be extracted and fused
        """

        # Loop through RGB-D images and fuse them together
        i = i + self.start
        print("Fusing frame (%d) %d/%d" % (i, i - self.start, self.n_imgs))

        try:
            color_img_arr, class_img_arr, depth_img_arr, cam_pose = self.preprocessor.get_images(i, use_masks)
        except (IOError, FileNotFoundError) as e:
            print('Warning: file ', i, ' missing. It was skipped', e)
            return

        # Skips this image if camera pose contains nan or inf values.
        if np.isnan(cam_pose).any() or np.isinf(cam_pose).any():
            print('Warning: camera pose ', i, ' is invalid. It was skipped')
            return

        if "debug" in params and params["debug"]["export_class_image"]["enabled"]:
            cp = class_img_arr.copy()
            if params["debug"]["export_class_image"]["ignore_score"]:
                cp[:, :, 2] = 0
                
            cv2.imwrite(params["debug"]["export_class_image"]["path"] + "/%05d_class_img.png" % i, cp*3)
        # Integrate observation into voxel volume (assume color aligned with depth)

        self.tsdf_vol.integrate(color_img_arr, depth_img_arr, class_img_arr, self.cam_intr, cam_pose, self.classes,
                                obs_weight=1.)

        if self.preprocessor.critical_instance_count():  # or i % (fuse_instance_interval+1) == fuse_instance_interval:
            print("--------------------")
            print("instance space running low!!!!!!!!")
            print("going to combine instances")
            print("----------------------------------------------------    ")
            hit_count_low = params["instance_generation"]["hit_count_low"]
            self.combine_instances(threshold=100, hit_count = hit_count_low, remove_small_instances=True)
            self.preprocessor.instance_generator.critical_instance_count = False

    def get_connected_components(self, instance_map, hit_count = 100):
        """
         returns a list with lists of ids that are connected.
         e.g. ids 1,2,3 and 4,5 are connected -> [[1,2,3],[4,5]
        @param instance_map: adjacency matrix that contains values of how many times these ids touched. e.g. [0,10;10;0] means ID 0 and 1 touched 10 times
        @param hit_count: how many times IDs have to touch each other in order to be counted as connected
        @return: list of list of ids. [[1,2,3], [4,5]]
        """
        mask = np.where(instance_map != 0)[0]
        if np.any(mask):
            max = np.max(mask) + 1
            graph = nx.convert_matrix.from_numpy_matrix(instance_map[0:max, 0:max] > hit_count)
            return [np.fromiter(l, int, len(l))+1 for l in nx.connected_components(graph) if len(l) > 1]
        return []


    def combine_instances(self, threshold = 0, hit_count = 100, instance_size_threshold = 1000, remove_small_instances = True):
        """
        Combines instances in the Semantic Instance volume if they have touched enough times
        @param threshold: Only check if instances need to be combined if there are more than threshold instances.
        @param hit_count: Only merge instances that have touched at least "hit_count" times
        """
        print("------------------- Combining instances ----------------------")
        instance_map = self.tsdf_vol.get_instance_mapping()
        classes = self.preprocessor.get_used_classes()
        self.preprocessor.instance_generator.print_instance_count()

        inst_count = self.preprocessor.instance_generator.get_instance_count()
        with self.tsdf_vol as volume:
            start = time.time()
            dict = {}
            for _c in classes:
                if inst_count[_c] > threshold:
                    print("Combining class", _c)
                    connected = self.get_connected_components(instance_map[_c, :, :], hit_count=hit_count)
                    dict[_c] = connected
                    #for i in connected:
                    #    self.preprocessor.combine_instances(_c, i[0], i[1:])
            real_inst = volume.combine_instances(dict, instance_map, instance_size_threshold, remove_small_instances)

            print("Combining took: ", time.time() - start)
            # for _c in real_inst.keys():
            #     self.preprocessor.instance_generator.remove_id()

            #removes all empty instances:
            for _class in real_inst.keys():
                fake_inst = self.preprocessor.instance_generator.get_unique_instances(_class)
                for f in fake_inst:
                    if f not in real_inst[_class]:
                        self.preprocessor.instance_generator.remove_id(_class, f)

        self.preprocessor.instance_generator.print_instance_count()


    def get_volume_extractor(self, mapping, color_dict):
        """
        Returns a volume extractor object for the semantic instace volume
        @param mapping: mapping class id -> name
        @param color_dict: mapping class id -> color
        @return: volume extractor to extract meshes
        """
        return VolumeExtractor(self.tsdf_vol, mapping, color_dict)
