# Semantic Instance Fusion in Python
This is a python script that fuses RGB-D images and their poses into a Truncated Signed Distance function that additionally contains semantic and instance information for each voxel. 

Once the Semantic Instance TSDF Volume is fused, different meshes (Instance, Semantic, RGB, Score) can be extracted.
<img src="https://github.com/renezurbruegg/SemanticInstanceFusion/raw/master/imgs/example.png" width="450">


<img src="https://github.com/renezurbruegg/SemanticInstanceFusion/raw/master/imgs/s3_overview.PNG" width="300"><img src="https://github.com/renezurbruegg/SemanticInstanceFusion/raw/master/imgs/scene3_instance.PNG" width="300">

## References
This code is based on the TSDF Fusion implementation by [Andy Zeng](https://github.com/andyzeng/tsdf-fusion-python).

## Installation
Make sure to have a working cuda version installed.
```
pip install -r requirements.txt
```
If you want to extracts masks for RGB images install pytorch
```
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
```
An download our pretrained Model
```
python src\MaskRCNNImpl\Models\download_model.py
```
## Fusing a custom dataset
In order to fuse a custom dataset, the following files have to be provided:
1. RGB images, 640x480
2. Depth images, 640x480
3. Pose Information for each image
4. Camera intrinsics

If your dataset is stored somewhere else, you can modify the path parameters in the params.yaml file.
```yaml
# specify path to the backbone model for mask RCNN
model_path: "src/MaskRCNNImpl/Models/Resnet_40_trained.pth"
# specify path to labels mapping. 
# This is a csv file that defines what semantic classes should be extracted and how they should be colored
# class_id,class_name,R,G,B
labels_path: "Dataset/labels_40.csv"
# Path to camera instrinsic matrix (4x4)
camera_intrinsics_path : "Dataset/camera-intrinsics.txt"
# Path to the RGB images
rgb_path: "Dataset/color"
# Path to the depth images
d_path: "Dataset/depth"
# Path to the pose files for each image
pose_path: "Dataset/pose"
# Path where meshes should be extracted to
output_path: "Dataset/output"

# Path to folder where semantic masks should be stored (if they even should be storeD)
mask_path: "Dataset/masks"

```
 
**Note:** Color images are saved as 24-bit PNG RGB, depth images are saved as 16-bit PNG in millimeters.
## Getting Started
### Extract a mesh for a given dataset
Simple run 
```
python main.py
```
All extracted meshes will be stored in the 'Dataset/output' folder.

### How it works
#### Fusion
![Pipeline](https://github.com/renezurbruegg/SemanticInstanceFusion/raw/master/imgs/pipeline1.PNG)

For each RGB-D image + pose a semantic mask is extracted using a pretrained Mask-RCNN model. 
These masks are than assigned a instance ID from the instance generator, that should be consistent through time.
These masks and the RGB-D images are then fused into a TSDF volume that additionally contains Semantic and Instance information for each voxel.
Since Masks from the Mask RCNN sometimes get different instance IDs, we combine all instances together, that have touched for at least 'hit_count' times.

Every voxel has a score value, that describes how much we trust in the given semantic class allocation.
#### Extraction
![Pipeline](https://github.com/renezurbruegg/SemanticInstanceFusion/raw/master/imgs/pipeline2.PNG)

We use marching cubes to extract a mesh from a tsdf volume. Each vertice is colored according to either the RGB values, Semantic Class, Instance number or score.

### params.yaml
The params file lets you specify most of the parameters used during the fusion process.
A description of each parameter is given in the comments of the params file.

### Output
Open output files with [MeshLab](http://www.meshlab.net/]), to see the output.


#### Q&A
 > I want to use a custom segmentation network:

  Just change the 'get_mask_for_images' function in the *'src/SemanticMasks/segmenter.py'* class
