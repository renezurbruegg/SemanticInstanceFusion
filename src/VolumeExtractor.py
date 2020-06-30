import colorsys
import random
import time
from enum import Enum

import numpy as np
import trimesh
from skimage import measure

usePC = True

#############################################################################################
#             Extracts a 3D Mesh from an Semantic Instance Volume                           #
#############################################################################################

class Color_mode(Enum):
    SEMANTIC = 1
    SCORE = 2
    ORIGINAL = 3
    INSTANCE = 4
    INSTANCE_EVALUATION = 5

class InstanceNotFoundException(Exception):
    pass


class VolumeExtractor:
    def __init__(self, volume, mapping, color_dict):
        """
        Creates a new volume extractor object
        @param volume: the SemInstVolume that should be extracted
        @param mapping: mapping: (class number -> name)
        @param color_dict: color mapping (class_number -> color)
        """
        self.volume = volume
        self.extracted_vol = None
        self.mapping = mapping
        self.color_dict = color_dict

    def get_volume(self):
        """
        Returns a cached VolumeMesh
        @return: VolumeMesh
        """
        if self.extracted_vol is None:
            self.extracted_vol = VolumeMesh(self.volume, self.color_dict)
        return self.extracted_vol


    def extract_mesh(self, colormode = Color_mode.SEMANTIC, instance_generator = None):
        """
        Extracts a mesh
        @param colormode: colormode to use
        @param instance_generator: instance generator that was used to assign instances
        @return: mesh (vertices, faces, norms, colors)
        """
        meshVol = self.get_volume()
        return meshVol.verts, meshVol.faces, meshVol.norms, meshVol.getColorsForMode(colormode, instance_generator)

    def extract_class(self, _class, colormode = Color_mode.INSTANCE):
        """
        Extracts the mesh for a given class
        @param _class: the class to extract
        @param colormode: colormode to use
        @return: mesh (vertices, faces, norms, colors)
        """
        meshVol = self.get_volume()

        return meshVol.extract_instance(_class, colormode)


    def extract_instances(self, _class, colormode = Color_mode.INSTANCE):
        """
        Extracts all instnaces for a given class
        @param _class:
        @param colormode:
        @return: list with meshes [mesh (vertices, faces, norms, colors)]
        """
        meshVol = self.get_volume()

        return meshVol.extract_instances(_class, colormode)

    def get_avaiable_classes(self):
        return self.get_volume().get_avaiable_classes()


class VolumeMesh:

    def getColorForClass(self, classNumb):
        """

        @param classNumb: class number
        @return: unique color for this class number or #fff if not avaiable
        """
        return self.color_dict.get(int(classNumb), (255, 255, 255))

    def getColorForScore(self, score):
        """
        Returns a color for this score
        High score -> Red, Low Score -> Blue
        @param score: score of current voxel
        @return: color for this score
        """
        # score in 0->255
        color = colorsys.hsv_to_rgb(2/3 + 1/3 * score/255, 1, 1)
        return (255*color[0], 255*color[1], 255*color[2])

    def getColorForInstance(self, i, size):
        """
        Returns a unique color for an instance
        @param i: instanceId that should get a color
        @param size: amount of instances that will be assigned a color in total
        @return: color
        """
        color = colorsys.hsv_to_rgb(i/(size+1), 1, 1)
        return (255*color[0], 255*color[1], 255*color[2])

    def getColorForInstanceAndClass(self, i, c):
        """
        Returns a color with instance as R and class as G value
        @param i: instance number
        @param c: class number
        @return: color
        """
        return (i, c, 0)


    def __init__(self, volume, color_dict):
        """
        Creates a new Volume mesh
        @param volume:  SemInstVolume from which meshes should be extracted
        @param color_dict: A color mapping (semantic_class -> RGB Color)
        """
        print("extracting volume")
        self.tsdf_vol, self.color_vol, self.class_vol = volume.get_volume()
        self.color_dict = color_dict
        self.volume = volume

        self.scores_raw = np.floor(self.class_vol / (256 * 256))
        self.class_numbers_raw = np.floor((self.class_vol - self.scores_raw * (256 * 256)) / 256).astype(int)
        self.instances_raw = (self.class_vol - self.scores_raw * 256 * 256 - self.class_numbers_raw * 256).astype(int)

        self.unique_numbers = np.unique(self.class_numbers_raw)
        t = time.time()
        verts, faces, norms, vals = measure.marching_cubes_lewiner(self.tsdf_vol, level=0, step_size=1)
        print("marching cubes took", time.time() - t)

        self.faces = faces
        self.norms = norms
        self.vals = vals

        self.verts_ind = np.round(verts).astype(int)
        self.verts = verts * volume.get_voxel_size() + volume.get_origin()
        self.verts_orig = verts

        class_label_score = self.class_vol[self.verts_ind[:, 0], self.verts_ind[:, 1], self.verts_ind[:, 2]]

        self.scores = np.floor(class_label_score / (256 * 256))
        self.class_numbers = np.floor((class_label_score - self.scores * (256 * 256)) / 256).astype(int)
        self.instances = (class_label_score - self.scores * 256 * 256 - self.class_numbers * 256).astype(int)
        self.unique_numbers = np.unique(self.class_numbers)
        
        rgb_vals = self.color_vol[self.verts_ind[:, 0], self.verts_ind[:, 1], self.verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / (256 * 256))
        colors_g = np.floor((rgb_vals - colors_b * (256 * 256)) / 256)
        colors_r = rgb_vals - colors_b * (256 * 256)- colors_g * 256

        self.colors_orig = np.floor(np.asarray([colors_r, colors_g, colors_b])).T.astype(np.uint8)

        self.colors_semantic = np.array([self.getColorForClass(c) for c in self.class_numbers]).astype(np.uint8)

        self.colors_score = np.array([self.getColorForScore(s) for s in self.scores]).astype(np.uint8)


    def getColorsForMode(self, colormode, instance_generator = None):
        """
        Return a color for each vertice of the mesh for the volume
        @param colormode: Color_mode
        @param instance_generator:  the instance generator that was used to assign instances
        @return: numpy array with color for each vertice
        """
        if colormode == Color_mode.SEMANTIC:
            return self.colors_semantic
        elif colormode == Color_mode.SCORE:
            return self.colors_score
        elif colormode == Color_mode.ORIGINAL:
            return self.colors_orig
        elif colormode == Color_mode.INSTANCE:
            if instance_generator is None:
                raise ValueError("need instance generator to create color for instance")

            return np.array(
                [self.getColorForInstance(*instance_generator.get_unique_id(s[0], s[1])) for s in
                 np.vstack([self.class_numbers, self.instances]).T]).astype(np.uint8)

        elif colormode == Color_mode.INSTANCE_EVALUATION:
            if instance_generator is None:
                raise ValueError("need instance generator to create color for instance")

            return np.array(
                [self.getColorForInstanceAndClass(s[0], s[1]) for s in
                 np.vstack([self.class_numbers, self.instances]).T]).astype(np.uint8)

        raise ValueError("Color mode not supported")

    def get_avaiable_classes(self):
        """
        @return: numpy array with all class numbers in this volume
        """
        return self.unique_numbers

    def extract_instance(self, class_number, colormode):
        """
        Returns the mesh for a given class
        @param class_number: class to extract
        @param colormode: colormode to use
        @return: mesh (vertice, faces, norms, colors)
        """
        if class_number not in self.unique_numbers:
            raise InstanceNotFoundException()

        tsdf_vol_cpy = self.tsdf_vol.copy()
        tsdf_vol_cpy[self.class_numbers_raw != class_number] = 1
        tsdf_vol_cpy[self.instances_raw == 0] = 1

        t = time.time()
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol_cpy, level=0)
        verts_ind = np.round(verts).astype(int)
        print("marching cubes took", time.time() - t)
        col = None
        if colormode == colormode.SCORE:
            score_vals = self.scores_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            col = np.array([self.getColorForScore(s) for s in score_vals]).astype(np.uint8)

        elif colormode == colormode.INSTANCE:
            instance_vals = self.instances_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            unique_instances = np.unique(instance_vals)
            print("instances:", unique_instances)
            lookup = np.zeros((np.max(unique_instances) + 1,))
            lookup[unique_instances] = np.arange(1, unique_instances.size + 1)
            #unique_instances.size
            col = np.array([self.getColorForInstance(lookup[i], unique_instances.size) for i in instance_vals]).astype(np.uint8)

        elif colormode == colormode.SEMANTIC:
            class_numbers = self.class_numbers_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
            col = np.array([self.getColorForClass(c) for c in class_numbers]).astype(np.uint8)

        if col is None:
            raise ValueError("Color mode not supported")


        return verts * self.volume.get_voxel_size() + self.volume.get_origin(), faces, norms, col


    def extract_instances(self, class_number, colormode):
        """
        Extracts a mesh for each instance
        @param class_number: semantic class to extract
        @param colormode: colormode
        @return: list containing meshes [ (vertices, faces, norms, colors), (vertices, faces, norms, colors)]
        """
        if class_number not in self.unique_numbers:
            raise InstanceNotFoundException()

        tsdf_vol_cpy2 = self.tsdf_vol.copy()
        tsdf_vol_cpy2[self.class_numbers_raw != class_number] = 1
        tsdf_vol_cpy2[self.instances_raw == 0] = 1

        list = []
        for i in np.unique(self.instances_raw[self.class_numbers_raw == class_number]):
            try:
                tsdf_vol_cpy = tsdf_vol_cpy2.copy()
                tsdf_vol_cpy[self.instances_raw != i] = 1
                t = time.time()
                verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol_cpy, level=0)
                verts_ind = np.round(verts).astype(int)
                print("marching cubes took", time.time() - t)
                col = None
                if colormode == colormode.SCORE:
                    score_vals = self.scores_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
                    col = np.array([self.getColorForScore(s) for s in score_vals]).astype(np.uint8)

                elif colormode == colormode.INSTANCE:
                    instance_vals = self.instances_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
                    unique_instances = np.unique(instance_vals)
                    print("instances:", unique_instances)
                    lookup = np.zeros((np.max(unique_instances) + 1,))
                    lookup[unique_instances] = np.arange(1, unique_instances.size + 1)
                    #unique_instances.size
                    col = np.array([self.getColorForInstance(i, unique_instances.size) for i in instance_vals]).astype(np.uint8)


                elif colormode == colormode.SEMANTIC:
                    class_numbers = self.class_numbers_raw[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
                    col = np.array([self.getColorForClass(c) for c in class_numbers]).astype(np.uint8)

                if col is None:
                    raise ValueError("Color mode not supported")

                list.append((verts * self.volume.get_voxel_size() + self.volume.get_origin(), faces, norms, col))
            except ValueError as e:
                print("got value error",e)
        return list


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)


    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

