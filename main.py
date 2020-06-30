import time

from src.SemanticMasks.mask_generators import DynamicMaskGenerator, MaskFromFileGenerator, CachedMaskGenerator, \
    MaskFromGroundtruthGenerator
from src.VolumeExtractor import Color_mode, pcwrite

try:
    import src.MaskRCNNImpl.Model as Model
except Exception as e:
    print("could not import Model", e)

from src import SemanticFuser
import yaml
with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

from src.VolumeExtractor import meshwrite
import csv


def get_classes_to_extract(classes):
    """ flips dictionary -> used to get the classes which are supposed to be extracted"""
    dict = {}
    for key in classes.keys():
        dict[classes[key]] = int(key)
    return dict

def get_classes(path):
    """ Gets mapping class number <-> class label"""

    with open(path, newline='') as csvfile:
        labelreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        labels = {}
        mapping = {}
        color_dict = {}
        for row in labelreader:
            labels[row[0]] = row[1]
            mapping[row[1]] = int(row[0])
            color_dict[int(row[0])] = (int(row[2]), int(row[3]), int(row[4]))
    return labels, mapping, color_dict

########################

### parameters



segment_mode = params["segment_mode"]
save_masks = params["semantic_masks"]["save_masks"]
model_path = params["model_path"]
mask_path = params["mask_path"]
labels_path = params["labels_path"]
prefetch_size = params["semantic_masks"]["prefetch_size"]
mask_threshold = params["semantic_masks"]["mask_threshold"]
gt_instance_path = params["gt_instance_path"]
gt_label_path = params["gt_label_path"]
img_size = params["img_size"]
n_start = params["n_start"]
rgb_path = params["rgb_path"]
d_path = params["d_path"]
pose_path = params["pose_path"]
score_threshold = params["semantic_masks"]["score_threshold"]
voxel_size = params["voxel_size"]
filter_interval = params["filter_interval"]
use_depth = params["semantic_masks"]["use_depth"]
depth_f = params["semantic_masks"]["depth_f"]
use_sp = params["semantic_masks"]["use_sp"]
fast_slic_comp = params["semantic_masks"]["fast_slic_comp"]
rcnn_threshold = params["semantic_masks"]["rcnn_threshold"]
rcnn_threshold_2 = params["semantic_masks"]["rcnn_threshold_2"]
skip = params["skip"]
n_images = params["n_images"]

combine_instances = params["instance_generation"]["combine_instances"]
hit_count = params["instance_generation"]["hit_count"]
instance_threshold = params["instance_generation"]["instance_threshold"]

output_path = params["output_path"]


# Debug Parameters
monitor_progress = params["debug"]["monitor_progress"]["enabled"]
monitor_progress_path = params["debug"]["monitor_progress"]["path"]
progress_extract_volume_images = params["debug"]["monitor_progress"]["volume_images"]
extract_instance_meshes = params["debug"]["extract_meshes_for_instance"]["enabled"]
class_to_extract_instances = params["debug"]["extract_meshes_for_instance"]["class"]
monitor_instances = params["debug"]["monitor_progress"]["monitor_instances"]["enabled"]
monitor_instances_combine = params["debug"]["monitor_progress"]["monitor_instances"]["combine"]
classes_to_monitor = params["debug"]["monitor_progress"]["monitor_instances"]["classes"]


if __name__ == '__main__':

    classes, mapping, color_dict = get_classes(labels_path)
    classes_to_extract = get_classes_to_extract(classes)

    segment_mode = segment_mode
#different segment modes:
    if segment_mode == 0:
        device = Model.get_device()
        mask_generator = DynamicMaskGenerator(rgb_path, model_path, img_size, device,
                                              Save_Flag = save_masks, mask_path= mask_path, labels_path= labels_path,
                                              prefetch_size = prefetch_size, score_threshold = score_threshold,
                                              mask_threshold= mask_threshold, skip = skip)
    elif segment_mode == 1:
        mask_generator = MaskFromFileGenerator(mask_path)
    elif segment_mode == 2:
        device = Model.get_device()
        mask_generator = CachedMaskGenerator(rgb_path, model_path, mask_path, img_size, device, labels_path,
                                             prefetch_size=prefetch_size,
                                             score_threshold = score_threshold, mask_threshold= mask_threshold, skip = skip)
    elif segment_mode == 3:
        mask_generator = MaskFromGroundtruthGenerator(gt_instance_path, gt_label_path, img_size)

    start_time = time.time()



    fuser = SemanticFuser.SemanticFuser(classes,
                                        n_imgs=n_images,
                                        start = n_start,
                                        path_to_rgb= rgb_path,
                                        path_to_d= d_path,
                                        path_to_pose= pose_path,
                                        score_threshold = score_threshold,
                                        voxel_size= voxel_size,
                                        filter_interval = filter_interval,
                                        mask_generator = mask_generator,
                                        use_depth= use_depth,
                                        depth_f = depth_f,
                                        use_sp= use_sp,
                                        fast_slic_comp=fast_slic_comp,
                                        instance_threshold=instance_threshold,
                                        rcnn_threshold= rcnn_threshold,
                                        rcnn_threshold_2= rcnn_threshold_2)

    for i in range(n_images):
        fuser.fuse_images(i, i % skip == 0)
        if monitor_progress:
            output_path =monitor_progress_path

            if progress_extract_volume_images:
                extractor = fuser.get_volume_extractor(mapping, color_dict)
                meshwrite(output_path + "/volume_instance_f"+str(i)+".ply", *extractor.extract_mesh(Color_mode.INSTANCE, instance_generator=fuser.preprocessor.instance_generator))
                meshwrite(output_path + "/volume_semantic_f"+str(i)+".ply", *extractor.extract_mesh(Color_mode.SEMANTIC))
                meshwrite(output_path + "/volume_score_f"+str(i)+".ply", *extractor.extract_mesh(Color_mode.SCORE))
                meshwrite(output_path + "/volume_original_f"+str(i)+".ply", *extractor.extract_mesh(Color_mode.ORIGINAL))

            if monitor_instances:
                c =  classes_to_monitor
                extractor = fuser.get_volume_extractor(mapping, color_dict)
                for _class in extractor.get_avaiable_classes():
                    if c == -1 or c ==_class:
                        try:
                            verts, faces, norms, colors = extractor.extract_class(_class)
                            meshwrite(output_path + "/" + classes[str(_class)] + "_f"+str(i)+".ply", verts, faces, norms, colors)

                            cnt = 0
                            import os

                            if not os.path.exists(output_path + "/instances_frame_" + str(i)):
                                os.mkdir(output_path + "/instances_frame_" + str(i))

                            for args in extractor.extract_instances(_class, Color_mode.SEMANTIC):

                                meshwrite(output_path + "/instances_frame_"+str(i)+"/" + classes[str(_class)] + "instance_f"+str(i)+"_"+str(cnt)+".ply", *args)
                                cnt += 1
                        except ValueError as e:
                            print("Couldn't export mesh: ", e)

            if monitor_instances and monitor_instances_combine:

                fuser.combine_instances(threshold=0, hit_count=hit_count)

                if progress_extract_volume_images:
                    extractor = fuser.get_volume_extractor(mapping, color_dict)
                    meshwrite(output_path + "/volume_instance_combined_f" + str(i) + ".ply",
                              *extractor.extract_mesh(Color_mode.INSTANCE,
                                                      instance_generator=fuser.preprocessor.instance_generator))
                    meshwrite(output_path + "/volume_semantic_combined_f" + str(i) + ".ply",
                              *extractor.extract_mesh(Color_mode.SEMANTIC))
                    meshwrite(output_path + "/volume_score_combined_f" + str(i) + ".ply",
                              *extractor.extract_mesh(Color_mode.SCORE))
                    meshwrite(output_path + "/volume_original_combined_f" + str(i) + ".ply",
                              *extractor.extract_mesh(Color_mode.ORIGINAL))

                c = classes_to_monitor
                extractor = fuser.get_volume_extractor(mapping, color_dict)
                for _class in extractor.get_avaiable_classes():
                    if c == -1 or c == _class:
                        try:
                            verts, faces, norms, colors = extractor.extract_class(_class)
                            meshwrite(output_path + "/" + classes[str(_class)] + "_combined_f" + str(i) + ".ply", verts, faces,
                                      norms, colors)

                            import os
                            cnt = 0

                            if not os.path.exists(output_path + "/instances_fused_frame_" + str(i)):
                                os.mkdir(output_path + "/instances_fused_frame_" + str(i))
                            for args in extractor.extract_instances(_class, Color_mode.SEMANTIC):
                                meshwrite(output_path + "/instances_fused_frame_" + str(i) + "/" + classes[
                                    str(_class)] + "instance_f" + str(i) + "_" + str(cnt) + ".ply", *args)
                                cnt += 1
                        except ValueError as e:
                            print("Couldn't export mesh: ", e)




    time_diff = time.time()-start_time
    print("fused images in: " + str(time_diff))
    print("FPS: ", n_images/time_diff)
    start_time = time.time()

    if combine_instances:
        fuser.combine_instances(hit_count = hit_count)

    extractor = fuser.get_volume_extractor(mapping, color_dict)
    output_path = output_path

    meshwrite(output_path + "/volume_instance.ply", *extractor.extract_mesh(Color_mode.INSTANCE, instance_generator= fuser.preprocessor.instance_generator))
    meshwrite(output_path + "/volume_semantic.ply", *extractor.extract_mesh(Color_mode.SEMANTIC))
    meshwrite(output_path + "/volume_score.ply", *extractor.extract_mesh(Color_mode.SCORE))
    meshwrite(output_path + "/volume_original.ply", *extractor.extract_mesh(Color_mode.ORIGINAL))
    meshwrite(output_path + "/volume_evaluation.ply", *extractor.extract_mesh(Color_mode.INSTANCE_EVALUATION, instance_generator= fuser.preprocessor.instance_generator))


    for _class in extractor.get_avaiable_classes():
        try:
            print("Extracting ", classes[str(_class)])
            t1 = time.time()
            verts, faces, norms, colors = extractor.extract_class(_class)
            meshwrite(output_path +"/" + classes[str(_class)]+".ply",   verts, faces, norms, colors)
            print("instance extraction(rene) took", time.time() - t1)

            #  -----------------------------------------------------
            #   UNCOMMENT THIS TO EXPORT CoG OF INSTANCES AS .PLY
            #  -----------------------------------------------------

            # import numpy as np
            # cp = fuser.preprocessor.instance_generator.get_center_point_for_class(_class)
            # ind = np.where(cp[:, 0] != 999)[0]
            # import colorsys
            #
            #
            # def getColorForInstance(i, size):
            #     """
            #     Returns a unique color for an instance
            #     @param i: instanceId that should get a color
            #     @param size: amount of instances that will be assigned a color in total
            #     @return: color
            #     """
            #     color = colorsys.hsv_to_rgb(i / (size + 1), 1, 1)
            #     return (255 * color[0], 255 * color[1], 255 * color[2])
            #
            # color = cp[ind]*100
            # for i,c in enumerate(cp[ind]):
            #     col = getColorForInstance(i+1, cp[ind].shape[0])
            #     color[i,:] = np.array([col[0],col[1],col[2]])
            # pcwrite("pts_"+str(_class)+".ply", np.hstack([cp[ind] + np.array([0.0,1.1,-0.05]), color]))

            if extract_instance_meshes and class_to_extract_instances == _class:
                import os
                cnt = 0
                i = "end"
                if not os.path.exists(output_path + "/instances_fused_frame_" + str(i)):
                    os.mkdir(output_path + "/instances_fused_frame_" + str(i))

                for args in extractor.extract_instances(_class, Color_mode.SEMANTIC):
                    meshwrite(output_path + "/instances_fused_frame_" + str(i) + "/" + classes[
                        str(_class)] + "instance_f" + str(i) + "_" + str(cnt) + ".ply", *args)
                    cnt += 1

        except ValueError as e: print("Couldn't export mesh: ",e)
    fuser.preprocessor.instance_generator.print_instance_count()
