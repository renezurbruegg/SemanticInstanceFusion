import time

import numpy as np
from numba import njit, prange


usePC = True

import yaml
with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    import pycuda.driver as drv
    import pycuda.tools as tools
    import pycuda.autoinit


    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0

# Print debug informations
debug = params["debug"]["fusion"]["enabled"]
debug_classes = params["debug"]["fusion"]["classes"]
debug_instances= params["debug"]["fusion"]["instances"]
checkDuplicate = params["debug"]["fusion"]["duplicate_check"]

instance_mat_dtype = np.int # np.bool or np.int


#############################################################################################
#            Implementation of the Semantic Instance Volume and fusion algorithm            #
#############################################################################################


class ClassNotFoundError(Exception):
    pass


class TSDFVolume:

    def __init__(self, vol_bnds, voxel_size, filter_interval = -1, use_gpu=True):
        """
          @param vol_bnds: An ndarray of shape (3, 2). Specifies the xyz bounds (min/max) in meters.
          @param voxel_size: size of the voxel in meters
          @param filter_interval: how often to median filter the image. Not implemented anymore
          @param use_gpu: whether to use the gpu for the fusion process or not.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."


        self.filter_interval = filter_interval

        self.frame = 0
        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 2 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256
        self._class_const = 256 * 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2])
        )

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._class_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # self._score_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        # Matrix to keep track how many times instances touched
        self.instance_volume_cpu = np.zeros((41,256,256)).astype(instance_mat_dtype)

        # Copy voxel volumes to GPU
        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)

            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)

            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # Class information for each voxel
            self._class_vol_gpu = cuda.mem_alloc(self._class_vol_cpu.nbytes)
            cuda.memcpy_htod(self._class_vol_gpu, self._class_vol_cpu)

            # instance mapping table
            self.instance_volume_gpu = cuda.mem_alloc(self.instance_volume_cpu.nbytes)
            cuda.memcpy_htod(self.instance_volume_gpu, self.instance_volume_cpu)


            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
       __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * class_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im,
                                  float * class_im,
                                  bool * instance_volume) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          
          
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
              
              
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;

          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;

          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;

          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          
          
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          
          
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
          


           // Integrate classes
           float old_class = class_vol[voxel_idx];
           float old_score = floorf(old_class/(256*256));
          
           float new_class = class_im[pixel_y*im_w+pixel_x];
           float new_score = floorf(new_class/(256*256));
           
           //float a = 1.0;
           //float b = 0.0;//4.0;
           //float c = 0.0;//0.1;
           
           // method 1)
           //new_score = a * new_score + b* 1.0f/depth_value/depth_value + c * obs_weight/w_new;
           //new_score = floorf(fmax(fmin(new_score,255.0f),0.0f));
           //old_score = a * old_score + b* 1.0f/depth_value/depth_value + c * obs_weight/w_old;
           //old_score = floorf(fmax(fmin(old_score,255.0f),0.0f));
           
           // method 2)
           //new_score = a * 1.0f/depth_value * obs_weight/w_new * new_score;
           //new_score = floorf(fmax(fmin(new_score,255.0f),0.0f));
           
           int old_label = floorf((old_class-old_score*256*256)/256);
           int new_label = floorf((new_class-new_score*256*256)/256);

           int new_id = new_class - new_score*256*256 - new_label*256;
           int old_id = old_class - old_score*256*256 - old_label*256;
            if(new_label == 0.0f) 
                return;
                
            if(old_label == 0.0f || old_score <= 10.0f) {
            
            } else if(old_label == new_label) {
                new_score = fmin((old_score + new_score),255.0f);
                if (depth_diff > - trunc_margin/3 && old_id != 0.0 && new_id != old_id) {
                
                   // two different ids were mapped to two objects.
                   // Need to merge them later   
                   //printf("%d: %d\\n", threadIdx.x, instance_volume[0]);
                   
                    // index to select position in instance_volume. This works like a adjacency matrix for instances
                    int base_label = 256*256*new_label;
                    int indexab = base_label + 256*new_id + old_id - 257;
                    int indexba = base_label + 256*old_id + new_id - 257;
                    
                    // ============================================================================================================
                    // UNCOMMENT THIS TO ONLY FUSE INSTANCES THAT HAVE TOUCHED ONCE (MAYBE ALSO CHANGE TYPE OF INSTANCE MAP TO BOOL)
                    //
                    //instance_volume[indexab] = 1;
                    //instance_volume[indexba] = 1;
                    
                    
                    // ============================================================================================================
                    // UNCOMMENT THIS TO INCREASE TO TIMES INSTANCES TOUCHED (SLOWER - ATOMIC ADD)
                    //
                    
                    atomicAdd(((int*) instance_volume) + indexab,1);
                    atomicAdd(((int*) instance_volume) + indexba,1);
                }
                //new_id = old_id;
            } else {
                new_score = fmax((old_score - new_score), 0.0f);
                new_label = old_label;
                new_id = old_id;
            }
              
            class_vol[voxel_idx] = new_score*256*256+new_label*256 + new_id;
            return;
        }""")

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
            self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
                np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    def get_instance_mapping(self):
        """
        @return: a matrix showing how often which instances of which classed where mapped onto the same voxel
        """
        cuda.memcpy_dtoh(self.instance_volume_cpu, self.instance_volume_gpu)
        return self.instance_volume_cpu

    def filterClasses(self, class_vol, radius = 2):
        """
        Applies a filter on the class volume. NOT IMPLEMENTED ANYMORE
        @param class_vol: the class volume
        @param radius: filter size
        @return: filtered class volume
        """
        print("filter class volume")
        scores = np.floor(class_vol /self._color_const)
        class_numbers = np.floor((class_vol - scores * self._color_const) / 256)
        instances = class_vol - class_numbers*256 - scores*256*256
        # class_numbers = ndimage.filters.generic_filter(class_numbers, filter, footprint=morphology.ball(4))
        #class_numbers = ndimage.median_filter(class_numbers, footprint=morphology.ball(radius))
        print("numbers", np.unique(class_numbers))

        return class_numbers*256 + scores*self._color_const + instances

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates.
    """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates.
    """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix


    def filterVolume(self):
        """
        Filters the class volume. NOT IMPLEMENTED ANYMORE
        """
        cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
        self._class_vol_cpu = self.filterClasses(self._class_vol_cpu, 3).astype(np.float32)
        cuda.memcpy_htod(self._class_vol_gpu,  self._class_vol_cpu)

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
    """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, class_im, cam_intr, cam_pose, labels, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.
    Args:
        @param: color_im (ndarray): An RGB image of shape (H, W, 3).
        @param: depth_im (ndarray): A depth image of shape (H, W).
        @param: class_im (ndarray): A "RGB" image of shape (H,W,3). Containing the class number as B and the score as G value
        @param: cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        @param: cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        @param: labels (ndarray): Mapping class_number -> class name
        @param: obs_weight (float): The weight to assign for the current observation.
    """
        im_h, im_w = depth_im.shape
        if debug:
            new_classes = np.unique(class_im[..., 1])
            old_classes = new_classes

            if debug_instances:
                print("instances in image", np.unique(class_im[..., 0]))
                for c in new_classes:
                    m = class_im[..., 1] == c
                    print("Image: class", c, "instances", np.unique(np.unique(class_im[m, 0])))

            if (debug_classes):
                print("Current classes in image to fuse: ",  [labels.get(str(int(n))) for n in new_classes], " - ", new_classes)
                _, _, class_vol = self.get_volume()
                scores = np.floor(class_vol / self._color_const)
                class_numbers = np.floor((class_vol - np.floor(class_vol / self._color_const) * self._color_const) / 256)
                inst = class_vol - scores*256*256 - class_numbers*256
                print("instances in voxel storage", np.unique(inst))
                numbers = np.unique(class_numbers)

                if debug_instances:
                    for c in numbers:
                        m = class_numbers == c
                        print("Volume: class", c, "instances", np.unique(np.unique(inst[m])))

                old_classes = np.hstack([old_classes, numbers])

                print("Current classes in voxel storage (before fusing): ", [labels.get(str(int(n))) for n in numbers], " - ", numbers)


                for numb in numbers:
                    if(numb == 0):
                        continue
                    label = labels.get(str(int(numb)), "unknown")

                    class_mask = class_numbers == numb
                    #score = scores[class_mask] Changing score
                    score = scores[class_mask]
                    print("Class  " + label +" (", numb, ") #voxel: ", np.sum(score > 0),  " scores: ", np.round(np.unique(score)/255, decimals=2))


        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])

        # Fold class color image into a single channel image
        class_im = class_im.astype(np.float32)
        class_im = np.floor(class_im[..., 2] * self._color_const + class_im[..., 1] * 256 + class_im[..., 0])


        print("frame: ", self.frame)
        self.frame += 1
        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     self._class_vol_gpu,
                                     cuda.InOut(self._vol_dim.astype(np.float32)),
                                     cuda.InOut(self._vol_origin.astype(np.float32)),
                                     cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     cuda.InOut(np.asarray([
                                         gpu_loop_idx,
                                         self._voxel_size,
                                         im_h,
                                         im_w,
                                         self._trunc_margin,
                                         obs_weight
                                     ], np.float32)),
                                     cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                     cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     cuda.InOut(class_im.reshape(-1).astype(np.float32)),
                                     self.instance_volume_gpu,
                                     block=(self._max_gpu_threads_per_block, 1, 1),
                                     grid=(
                                         int(self._max_gpu_grid_dim[0]),
                                         int(self._max_gpu_grid_dim[1]),
                                         int(self._max_gpu_grid_dim[2]),
                                     )
                                     )
                if self.filter_interval != -1 and (self.frame % self.filter_interval) == (self.filter_interval -1):
                    self.filterVolume()


        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z
            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

            # Integrate class and scores
            class_numbers_score = self._class_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

            old_score = np.floor(class_numbers_score / self._color_const)
            old_label = np.floor((class_numbers_score - old_score * self._color_const) / 256)

            new_class_numbers_score = class_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_score = np.floor(new_class_numbers_score / self._color_const)
            new_label = np.floor((new_class_numbers_score - new_score * self._color_const) / 256)

            """

            if (old_label == 0.0f | | old_score == 0.0f) {

            } else if (new_label =! 0.0f && old_label == new_label    AND not (old_label == 0.0f | | old_score == 0.0f)) {
            new_score = fmin((old_score + new_score), 255.0f);
            } else if(new_label =! 0.0f AND not  old_label == new_label  AND not (old_label == 0.0f | | old_score == 0.0f)) {
            new_score = fmax((old_score - new_score), 0.0f);
            }

            class_vol[voxel_idx] = new_score*256*256+new_label*256;
                """

            zero = np.logical_or(old_label == 0, old_score == 0)
            skip_these = np.logical_or(old_label == 0, old_score == 0)
            not_skip_these = np.logical_not(skip_these)

            same_label = np.logical_and(np.logical_and(new_label == old_label, np.logical_not(zero)), not_skip_these)

            diff_label = np.logical_and(np.logical_and(new_label != old_label, np.logical_not(zero)), not_skip_these)

            new_score[same_label] = np.minimum(old_score[same_label] + new_score[same_label], 255)
            new_score[diff_label] = np.maximum(old_score[diff_label] - new_score[diff_label], 0)

            # new_label[skip_these] = old_label[skip_these]
            # new_score[skip_these] = old_score[skip_these]

            self._class_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_score * self._color_const + new_label * 256

        if debug:

            _, _, class_vol = self.get_volume()
            class_numbers = np.floor((class_vol - np.floor(class_vol / self._color_const) * self._color_const) / 256)
            new_classes_voxel = np.unique(class_numbers)
            if debug_classes:
                print("Current classes in voxel storage (after fusing): ", [labels.get(str(int(n))) for n in new_classes_voxel], " - ", new_classes_voxel)
                inst = class_vol - np.floor(class_vol / self._color_const) * 256 * 256 - class_numbers * 256
                print("instances in voxel storage AFTER", np.unique(inst))
            if checkDuplicate:
                for c in new_classes_voxel:
                    if c not in old_classes:
                        print("-------------------------------------")
                        print("----!!! found newly generated class !!!! -----")
                        print("----      "+ str(c) +  "  -----")
                        print("")
                        print("-------------------------------------")

    def get_voxel_size(self):
        return self._voxel_size

    def get_origin(self):
        return self._vol_origin

    def get_volume(self):
        if self.gpu_mode:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
            cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
        else:
            # Fix for cpu
            return self._tsdf_vol_cpu.copy(), self._color_vol_cpu.copy(), self._class_vol_cpu.copy()

        return self._tsdf_vol_cpu, self._color_vol_cpu, self._class_vol_cpu

    def __enter__(self):
        print("enter. detaching class label score")
        cuda.memcpy_dtoh(self._class_vol_cpu, self._class_vol_gpu)
        cuda.memcpy_dtoh(self.instance_volume_cpu, self.instance_volume_gpu)
        self._class_vol_cpu = self._class_vol_cpu.astype(np.float32)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._class_vol_cpu = self._class_vol_cpu.astype(np.float32)
        self.instance_volume_cpu = self.instance_volume_cpu.astype(instance_mat_dtype)

        cuda.memcpy_htod(self._class_vol_gpu,  self._class_vol_cpu)
        cuda.memcpy_htod(self.instance_volume_gpu,  self.instance_volume_cpu)
        print("exit. attaching class label score")

    def combine_instances(self, dict, instance_map, instance_size_threshold, remove_small_instances = True):
        """ replaces all occurences of rest with orig """
        scores = np.floor(self._class_vol_cpu /self._color_const) ###$$$$$$$$$
        class_numbers = np.floor((self._class_vol_cpu - scores * self._color_const) / 256) ###$$$$$$$$$
        instances = (self._class_vol_cpu - class_numbers*256 - scores*256*256).astype(int) ###$$$$$$$$$

        real_inst = {}

        for _class in dict.keys():
            # iterate through each class that contain instances that have to be combined
            mask = class_numbers == _class
            masked_inst = instances[mask]
            id_count = np.bincount(instances[mask], minlength=256)
            # count how many times each id occurs in the voxel volume
            for entry in dict[_class]:
                orig = entry[0]
                rest = entry[1:]

                instances[mask] = np.where(np.isin(masked_inst, rest), orig, instances[mask])

            if remove_small_instances:
                voxel_count_per_id = np.bincount(instances[mask])
                # count how many voxel per id there still are
                small_instances = np.where(np.logical_and(voxel_count_per_id > 0, voxel_count_per_id < instance_size_threshold))[0]

                for instance in small_instances:
                    # Replace all small instances with object they touched the most.
                    # If there is none, we will just assign instance id 0
                    most_touched_instance = np.argmax(instance_map[_class, instance] * (id_count > 0))

                    for ids in dict[_class]:
                        if most_touched_instance in ids:
                            most_touched_instance = ids[0]
                            break

                    instances[mask] = np.where(masked_inst == instance, most_touched_instance, instances[mask])

            # reset instance mapping
            real_inst[_class] = np.unique(instances[mask])

            self.instance_volume_cpu[_class,:,:] = np.zeros((256,256)).astype(instance_mat_dtype)

        self._class_vol_cpu = instances + class_numbers*256 + scores*256*256 ###$$$$$$$$$

        return real_inst


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
  """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
  """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts