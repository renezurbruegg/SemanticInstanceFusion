import numpy as np
import scipy.ndimage as ndi
import yaml
from scipy.spatial import distance

with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

debug = params["debug"]["instance_generator"]["enabled"]

#############################################################################################
#                     Assigns an instance ID for a Semantic Mask                            #
#############################################################################################

class InstanceGenerator:
    """
    Assigns an instance ID to a given semantic mask and class
    It stores the 3D center point of each instances and compares new instances with these points.
    If the instances are close together, they will be assigned the same instance id and the center point will be updated
    accordingly
    """
    def __init__(self, intrinsics_path, max_instances = 256, num_classes = 41):
        """
        Create a new instance generator
        @param intrinisc_path: path to the camera intrinsics (used to extract center point of a class)
        @param max_instances: Maximal amount of instances. Usually 255 since they will be stored in an image
        @param num_classes: How many different semantic classes exist.
        """
        self.no_instance = 999
        self.max_instances = max_instances
        self.num_classes = num_classes
        self.critical_instance_count = False
        self.instance_matrix = np.zeros((num_classes, max_instances, 3)) + self.no_instance
        self.intrinisc_inv = np.linalg.inv(np.loadtxt(intrinsics_path))
        self.class_inst_to_number_map = None
        self.class_inst_cnt = 0

    def combine_instances(self, _class, true_id, rest):
        """
        Replaces all occurences of the IDs specified in "rest" that belong to the class "_class" with the number "true_id"
        @param _class: Class whose instances should be combined
        @param true_id: Original instance ID
        @param rest: List of IDs that should be replaced with true_id.
        """
        #self.instance_matrix[_class, true_id, :] = self.instance_matrix[_class, np.max(rest), :]
        self.instance_matrix[_class, rest, :] = self.no_instance
        self.critical_instance_count = False

    def get_used_classes(self):
        """
        Returns a numpy array (1D) True/False for each class, to show if the class is present in the volume
        @return: 1d numpy array
        """
        return np.where(np.min(np.min(self.instance_matrix[:,:,:],axis = 1), axis = 1) != self.no_instance)[0]

    def clear(self):
        """ removes all instances"""
        self.instance_matrix = np.zeros((self.num_classes, self.max_instances, 3)) + self.no_instance

    def sync_instances(self, _class, ins_arr):
        """
        Makes sure that all instances for the class "_class" that don't match ins_arr are deleted
        @param _class: the class
        @param ins_arr: array containing instance number that should not be deleted
        """
        all_instances = np.arange(0, self.max_instances)
        # remove used instances from index
        all_instances = np.delete(all_instances-1, ins_arr)
        # remove all instances that are not in volume
        self.instance_matrix[_class, all_instances,:] = self.no_instance
        print("debug")

    def print_instance_count(self):
        """ prints how many instances are stored for each class"""
        _class = 0
        for cnt in self.get_instance_count():
            if cnt != 0:
                print("instances for class", _class, ": ", cnt)
            _class += 1

    def get_instance_count(self):
        """ returns how many instances are stored for each class as numpy array"""
        return np.sum((self.instance_matrix != 999),axis = 1)[:,0]

    def get_unique_instances(self,_class):
        return np.unique(np.where(self.instance_matrix[_class]!= self.no_instance))+1

    def plot_centers(self, _class):
        """ plots all stored centers for this class in an open3d window"""
        ins = self.instance_matrix[_class,:,:]
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        c = ins[ins != self.no_instance].reshape((-1, 3))
        if c.shape[0] == 1:
            c = np.vstack([np.array([0,0,0]), c])

        pcd.points = o3d.utility.Vector3dVector(c)
        o3d.visualization.draw_geometries([pcd])

    def get_unique_id(self, _class, instance_id):
        """
        returns a unique id for this _class and instance_id combination.
        E.g: _class, instance_id:
            0,1 -> 1, 0,2 -> 2, 1,1 -> 3, 1,2 -> 4 ...

        @param _class: the class for this instance_id
        @param instance_id:  the instance_id we want to map to a unique number
        @return: a unique number for each used (_class,instance_id) combination
        """
        if self.class_inst_to_number_map is None:
            self.class_inst_to_number_map = np.zeros((41,256)).astype(int)
            count = 0
            for i in range(self.num_classes):
                ins = self.instance_matrix[i, :, :]
                if np.any(ins != self.no_instance):
                    mapping = np.where(self.instance_matrix[i, :, 0] != 999)[0]
                    for j in mapping:
                        self.class_inst_to_number_map[i,j+1] = count
                        count += 1
            self.class_inst_cnt = count

        instances_for_this_class = self.class_inst_to_number_map[_class]
        real_inst = instances_for_this_class[instances_for_this_class != 0]
        if real_inst.size == 0:
            return 1,1
        return instances_for_this_class[instance_id] - np.min(real_inst) + 1, real_inst.size

    def remove_id(self, _class, inst):
        """
        Removes a given instance for given class from the generator
        @param _class: class of the instance
        @param inst: instance number
        """
        self.instance_matrix[_class,inst-1,:] = self.no_instance
        self.critical_instance_count = False

    #@njit(parallel=True)
    def closest_point(self, lookup_point, all_points):
        """ @param lookup_point: a n-dimensional point
            @paramall_points: an array with n-dimensional points
            @return the index of the node in the nodes array that is closes to the given point"""

        closest_index = distance.cdist([lookup_point], all_points).argmin()
        return closest_index

    #@njit(parallel=True)
    def get_center(self, mask):
        """
        Returns a 3D center point for this mask
        @param mask: the mask (pillow image)
        @return: 3D numpy array
        """
        m_arr = np.asarray(mask)
        center = ndi.center_of_mass(m_arr)
        return np.array([center[1], center[0]]).astype(int), np.sum(np.asarray(mask).astype(float))/(255*480*680)

    #@njit(parallel=True)
    def px_to_world(self, x, y, depth, ext):
        """
        convert image pixel and depth to 3d world coordinate
        @param x: x value of the pixel
        @param y: y value of the pixel
        @param depth: depth value for this pixel
        @param ext: Extrinsic of the camera
        @return: world coordinates of the given pixel
        """
        #world = np.matmul(np.linalg.inv(ext), np.hstack([depth * np.matmul(self.intrinisc_inv, np.array([x, y, 1])), 1]))
        x_coord = np.matmul(self.intrinisc_inv, np.array([x, y, 1]))*depth
        x_world = np.matmul(np.linalg.inv(ext[0:3,0:3]), (x_coord - ext[0:3,3]))
        return x_world


    def get_center_point_for_class(self,class_numb):
        return self.instance_matrix[class_numb, :, :]

    def get_id_for_class_mask(self, class_numb, mask, depth_image, pose, still_hot, threshold=0.5, factor=0.1):
        """
        Returns a unique ID for a class mask

        @param class_numb: Semantic class of this image
        @param mask: The mask for this image (pillow/numpy)
        @param depth_image: The depth image (numpy)
        @param pose: The pose (numpy)
        @param still_hot: List with instance_ids that where allready obtained for this image
        @param threshold: distance threshold, how far CoG of instances can be appart and still be interpreted as same instance
        @param factor:
        @return: unique id for this instance
        """
        instances = self.instance_matrix[class_numb, :, :]
        point, area_percentage = self.get_center(mask)
        area_percentage = min(max(0.05, area_percentage), 0.3)
        # scale threshold proportional to area of mask
        threshold = threshold * area_percentage/factor
        depth = depth_image[point[1], point[0]]
        if debug:
            if params["debug"]["instance_generator"]["class_number"] == -1 or params["debug"]["instance_generator"]["class_number"] == class_numb:
                print("occupied instances before for class", class_numb, ":")
                if np.where(instances != 999)[0].size != 0:
                    print(np.unique(np.where(instances != 999)[0]))
                else:
                    print("none")

        point = self.px_to_world(point[1], point[0], depth, pose)
        inst_id = self.closest_point(point, instances)
        # print(inst_id.shape)

        dist = np.linalg.norm(point - instances[inst_id, :], ord=2)

        if debug:
            print("instance ID", inst_id + 1, " was closest with dist:", dist)

        if dist > threshold or [class_numb, inst_id] in still_hot:
            if debug:
                print("instance too far apart. Going to create new instance")
            try:
                inst_id = np.nonzero(instances == self.no_instance)[0][0]
                #print("found unused id:", inst_id + 1)
                instances[inst_id, :] = point
            except:
                print("------------------- no unused id found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("THIS IS NO DRILL, THIS IS AN EMERGENCY!!!")
                print("FREE UP SPACE! I DO NOT CARE IF YOU HAVE TO THROW AWAY THE WHOLE TSDF VOLUME")

            if inst_id > 240:
                self.critical_instance_count = True
        else:
            #print("instance close enough. updating center of mass and returning id")
            # check if this combination is still hot.
            instances[inst_id, :] = (point + instances[inst_id, :]) / 2
        if debug:
            print("going to return:", inst_id + 1)
        return inst_id + 1, dist > threshold
