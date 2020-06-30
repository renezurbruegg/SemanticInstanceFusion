import csv
import os
import shutil
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


#####################################################################
# This file contains functions for loading/saving/testing the model #
#####################################################################
def get_paths_in(mypath):
    """returns paths of all jpgs in a directory"""
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.jpg')]


def rand_color(opacity):
    return (randint(0,255),randint(0,255),randint(0,255), opacity)
def get_classes(path):
    """reads the classes from a .csv file"""
    with open(path, newline='') as csvfile:
        labelreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        labels = {}
        for row in labelreader:
            labels[row[0]] = row[1]
    return labels
def get_40_classes(path="../nyu_v2_dataset/mapping/classNames40.txt"):
    """loads the class names for the nyu_v2_dataset"""
    f = open(path, "r")
    l = f.readline()
    items = l[:-2].split(",")
    i = 1
    dict = {}
    for item in items:
        dict[str(i)] = item
        i+=1

    print(dict)
    print(dict.__len__())
    return dict


#saves actually the whole model to the given path parameter
def save_weights(model, path = 'checkpoint.pth' ):
    """Saves model to a the given path. Has to be a .pth file."""
    torch.save(model, path)
#loads model from given path
def load_weights(path = 'checkpoint.pth', device = torch.device("cuda")):
    """Loads the weights of a saved model"""
    if (device == torch.device("cuda")):
        return torch.load(path)
    else:
        return torch.load(path, map_location={'cuda:0': 'cpu'})

def save_result(result, path, type="PNG", ext = ".png", mask_threshold = 0.5, score_threshold = 0.5):
    """Saves results of the segmentation to a given path with a given extension, will also apply thresholding"""
    masks = result['masks'].cpu().detach().numpy()
    labels = result['labels'].cpu().detach().numpy()
    boxes = result['boxes'].cpu().detach().numpy()
    scores = result['scores'].cpu().detach().numpy()
    
    
    for i in range(masks.shape[0]):
        if scores[i] > score_threshold:
            mask = np.zeros(masks[i, 0, :, :].shape).astype(np.uint8)
            mask[masks[i, 0, :, :] > mask_threshold] = 255
            mask = Image.fromarray(mask, mode='L')


            mask.save(path + "/" + str(i)+"_" + str(labels[i]) + "_" + str(scores[i])+ ext, type)
def load_result(path):
    """loads precomputed results from a given path"""
    masks = []
    labels = []
    scores = []
    for name in listdir(path):
        if not( "result" in name or "input" in name):
            print(name)
            mask = Image.open(path+ "/" + name)
            n = name.split("_")
            label = int(n[1])
            score = float(n[2][:-4])
            masks.append(mask)
            labels.append(label)
            scores.append(score)
    return masks, labels, scores

def get_result(model, img):
    """returns the segmentation of an img by a given model"""
    return model.forward(img)
#loads model but requires it to be in the same directory under the name model.pth
# def get_model(device = torch.device("cuda")):
#     if (device == torch.device("cuda")):
#         return torch.load("model.pth")
#     else:
#         return torch.load("model.pth", map_location={'cuda:0': 'cpu'})

#creates model with a bunch of tunable parameters
def get_model_instance_segmentation(num_classes, hidden_layer):
    """Creates a new Mask-R-CNN which has to be downloaded from torchvision (will not work on a cluster with
    restricted internet access)"""
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_device(specific = "None"):
    """If cuda is available this will return the cuda device, if cuda should not be used you can specify which device
    will be used with the parameter specific. Example: get_device(specific = "cpu")"""
    if(specific == "None"):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(specific)


#tests a given model by using a meme I found on my computer (just for debugging purposes)
#NOT A PROPER TEST FUNCTION (does not test masks and accuracy)
def test(model, device = torch.device('cpu'), max = 20, input_path = "../Test_Pictures/test.png", output_path = "resultat.jpg", classes = None):
    """test function that tests a model and saves the resulting image"""

    img = Image.open(input_path)
    txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt)
    input = img.copy().convert('RGBA')
    img = np.array(img).astype('float32') / 255.
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    img = [img]

    img = torch.as_tensor(img).to(device)
    model.eval()
    result = model.forward(img)[0]

    masks = result['masks'].cpu().detach().numpy()
    labels = result['labels'].cpu().detach().numpy()
    boxes = result['boxes'].cpu().detach().numpy()
    scores = result['scores'].cpu().detach().numpy()


    for i in range(min(masks.shape[0], max)):
        #ret, mask = cv2.threshold(masks[i, 0, :, :], 0.5, 1, cv2.THRESH_BINARY)
        label = labels[i]
        box = boxes[i].astype('int')
        #score = scores[i]
        d.rectangle((box[0],box[1],box[2],box[3]),outline = (255, 0, 0,255))
        if classes == None:
            d.text((box[0], box[1] + 5), str(label), fill=(0, 0, 0, 255))
        else:
            d.text((box[0], box[1] + 5), classes(label), fill=(0, 0, 0, 255))

    result = Image.alpha_composite(input,txt).convert("RGB")
    result.save(output_path, "JPEG")

def load_img(path,size):
    """loads image from a given path and resizes it into the given size.
    Returns the image as a numpy array (img) and as a RGBA image (input)"""
    img = Image.open(path)
    img = img.resize(size)
    # Input image needed in the end.
    input = img.copy().convert('RGBA')

    # Prepare input image for running through the model.
    img = np.array(img).astype('float32') / 255.
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img, input

def batch_load_img(pahts, size):
    """loads images from all given paths and resizes them accordingly"""
    imgs = []
    inputs = []

    for path in pahts:
        print("Image Path: " + path)
        img, input = load_img(path,size)
        imgs.append(img)
        inputs.append(input)

    return np.array(imgs), inputs
# def visualize_results(input, masks, output_path, type = "PNG",ext = ".png"):
#     box_image = Image.new('RGBA', input.size, (255, 255, 255, 0))
#     box_image_draw = ImageDraw.Draw(box_image)
#
#     # All pixels in the mask with a value bigger than this threshold are
#     # assigned to the binary mask.
#     mask_threshold = 0.5
#     # All detections with a score bigger than this threshold are used as actual
#     # detections.
#     score_threshold = 0.5
#
#     # Extract predicted boxes and masks.
#     for mask in masks:
#             mask = Image.fromarray(mask, mode='L')
#             box_image_draw.bitmap((0, 0), mask, fill=rand_color(160))
#
#     # Save the predicted test image.
#     out = Image.alpha_composite(input, box_image).convert("RGB")
#     out.save(output_path + ext, type)

def visualize(input, result, classes, output_path, type = "PNG", ext = ".png"):
    """Takes the input image, result, classes and saves a combined image to the output_path"""
    masks = result['masks'].cpu().detach().numpy()
    labels = result['labels'].cpu().detach().numpy()
    boxes = result['boxes'].cpu().detach().numpy()
    scores = result['scores'].cpu().detach().numpy()

    box_image = Image.new('RGBA', input.size, (255, 255, 255, 0))
    box_image_draw = ImageDraw.Draw(box_image)


    # All pixels in the mask with a value bigger than this threshold are
    # assigned to the binary mask.
    mask_threshold = 0.5
    # All detections with a score bigger than this threshold are used as actual
    # detections.
    score_threshold = 0.3

    # Extract predicted boxes and masks.
    for i in range(masks.shape[0]):
        if scores[i] > score_threshold:
            # Create binary object mask and draw it into the test image.
            mask = np.zeros(masks[i, 0, :, :].shape).astype(np.uint8)
            mask[masks[i, 0, :, :] > mask_threshold] = 255
            mask = Image.fromarray(mask, mode='L')
            box_image_draw.bitmap((0, 0), mask, fill=rand_color(160))

            # Draw the object bounding box.
            box = boxes[i].astype('int')
            box_image_draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0, 255))

            # Draw the class label.
            if classes == None:
                label = labels[i]
            else:
                label = classes[str(labels[i])]

            box_image_draw.text((box[0], box[1] + 5), str(label), fill=(0, 0, 0, 255))

    # Save the predicted test image.
    out = Image.alpha_composite(input, box_image).convert("RGB")
    out.save(output_path + ext, type)

def test_masks_v3(model, in_path, out_path, size = (640,480), device = torch.device('cpu'), classes = None, batch_size = 2, type = "PNG", ext = ".png"):
    """Takes model and two paths. It will segment all pictures in in_path and output them to out_path"""

    paths_in = get_paths_in(in_path)
    names = [f for f in listdir(in_path) if f.endswith('.jpg')]
    # Prediction for the test image.
    model.eval()
    for i in range(0, paths_in.__len__(), batch_size):
        print("batch nr: " + str(i))
        imgs, inputs = batch_load_img(paths_in[i:i+batch_size], size)
        imgs = torch.tensor(imgs).to(device)
        results =model.forward(imgs)
        for j in range(results.__len__()):
            name = names[i+j].split(".")[0]
            p = out_path + "/" + name
            if os.path.exists(p):
                shutil.rmtree(p)
            os.mkdir(p)
            print(name)

            visualize(inputs[j], results[j], classes,p +"/result", type = type, ext = ext)
            inputs[j].convert("RGB").save(p + "/input" + ext, type)
            save_result(results[j], p, type = type, ext = ext)
    del results
    del imgs
    del inputs
    torch.cuda.empty_cache()

# def test_masks_v2(model, paths_in, size = (640,480), device = torch.device('cpu'), out_name = "../Test_Results/result_", classes = None, batch_size = 2,type = "JPEG", ext = ".jpg"):
#     # Predict for the test image.
#     model.eval()
#
#     for i in range(0, paths_in.__len__(), batch_size):
#         print("batch nr: " + str(i))
#         imgs, inputs = batch_load_img(paths_in[i:i+batch_size], size)
#         imgs = torch.tensor(imgs).to(device)
#         results =model.forward(imgs)
#         for j in range(results.__len__()):
#             visualize(inputs[j], results[j], classes, out_name + str(i+j), type = type, ext = ext)
#     del results
#     del imgs
#     del inputs
#
# def test_masks(model, device = torch.device('cpu'), max = 100, input_path = "../Test_Pictures/test.jpg", output_path = "../Test_Results/resultat.jpg", test_batch = 1, classes = None):
#     # Load test image
#     img = Image.open(input_path)
#     input_size = img.size
#
#     # Input image needed in the end.
#     input = img.copy().convert('RGBA')
#
#     # Prepare input image for running through the model.
#     img = np.array(img).astype('float32') / 255.
#     img = np.swapaxes(img, 0, 2)
#     img = np.swapaxes(img, 1, 2)
#     temp = []
#     for i in range(test_batch):
#         temp.append(img)
#     img = np.array(temp)
#     img = torch.as_tensor(img).to(device)
#
#     # Predict for the test image.
#     model.eval()
#     start = time.time()
#     result = model.forward(img)[0]
#     print("forward: " + str(time.time()-start))
#     # Initialise container image for bounding boxes and class labels.
#     box_image = Image.new('RGBA', input_size, (255, 255, 255, 0))
#     box_image_draw = ImageDraw.Draw(box_image)
#
#     # Get predicted data.
#     masks = result['masks'].cpu().detach().numpy()
#     labels = result['labels'].cpu().detach().numpy()
#     boxes = result['boxes'].cpu().detach().numpy()
#     scores = result['scores'].cpu().detach().numpy()
#
#     # All pixels in the mask with a value bigger than this threshold are
#     # assigned to the binary mask.
#     mask_threshold = 0.5
#     # All detections with a score bigger than this threshold are used as actual
#     # detections.
#     score_threshold = 0.5
#
#     # Extract predicted boxes and masks.
#     for i in range(min(masks.shape[0], max)):
#         if scores[i] > score_threshold:
#             # Create binary object mask and draw it into the test image.
#             mask = np.zeros(masks[i,0,:,:].shape).astype(np.uint8)
#             mask[masks[i,0,:,:] > mask_threshold] = 255
#             mask = Image.fromarray(mask, mode='L')
#             box_image_draw.bitmap((0,0), mask, fill = rand_color(160))
#
#             # Draw the object bounding box.
#             box = boxes[i].astype('int')
#             box_image_draw.rectangle((box[0],box[1],box[2],box[3]),outline = (255, 0, 0,255))
#
#             # Draw the class label.
#             if classes == None:
#                 label = labels[i]
#             else:
#                 label = classes[str(labels[i])]
#
#             box_image_draw.text((box[0], box[1] + 5), str(label), fill=(0, 0, 0, 255))
#
#     # Save the predicted test image.
#     result = Image.alpha_composite(input, box_image).convert("RGB")
#     result.save(output_path, "JPEG")
#
#     del result
#     del img

def cleanup():
    """cleans memory"""
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
