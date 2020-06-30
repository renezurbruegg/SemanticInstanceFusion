import os

import numpy as np
import torch
from PIL import Image


class NYUV2Dataset(object):
    def __init__(self, root, num_labels, transforms):

        # Read the mapping from 894 to 40 classes.
        file = open(os.path.join(root, 'mapping', 'classMapping40.txt'), 'r')
        self.mapping40 = file.readline().replace('\n','').split(',')
        self.mapping40 = [int(num) for num in self.mapping40]
        file.close()

        # Read the mapping from 40 to 13 classes.
        file = open(os.path.join(root, 'mapping', 'classMapping13.txt'), 'r')
        self.mapping13 = file.readline().replace('\n','').split(',')
        self.mapping13 = [int(num) for num in self.mapping13]
        file.close()

        self.num_labels = num_labels - 1

        # Filepath to the NYU V2 dataset root folder.
        self.root = os.path.join(root, 'data')
        self.transforms = transforms
        # Loads all image filenames and sorts them in order to allign both folders.
        self.images = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.instances = list(sorted(os.listdir(os.path.join(self.root, "instances"))))
        self.labels = list(sorted(os.listdir(os.path.join(self.root, "labels"))))


    def __getitem__(self, idx):
        #print("Get image ", self.images[idx])

        # load images ad masks
        image_path = os.path.join(self.root, "images", self.images[idx])
        instance_path = os.path.join(self.root, "instances", self.instances[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        # Open images
        image = Image.open(image_path).convert("RGB")
        instance = np.array(Image.open(instance_path))
        label = np.array(Image.open(label_path))

        # Count the number of objects in the image, separate the instance numbers
        # of different labels and get the labels for each instance.
        instances_separated = np.zeros(instance.shape)
        labels_separated = []
        n_objects = 0
        for i in np.unique(label):
            # Considers only labels appearing in the image.
            if (i in label):
                # Add the number of instances of label i to the total number of existing instances.
                instances_separated[label == i] = instance[label == i] + n_objects
                # Appends the labels by the number of instances for this label.
                labels_separated = labels_separated + np.max(instance[label == i]) * [i]
                n_objects += np.max(instance[label == i])

        labels = np.array(labels_separated)
        instances_separated = instances_separated.astype(int)

        # Create a binary mask for every instance.
        masks = np.zeros((n_objects + 1, label.shape[0], label.shape[1]))
        for i in np.unique(instances_separated):
            masks[i, instances_separated == i] = 1

        # Remove mask 0 as this is the 'don't care' area.
        masks = masks[1:]

        # Removes too small masks and corresponding labels.
        thr = 10
        for i in range(len(masks)-1, -1, -1):
            sum_local = np.sum(masks[i,:,:])
            if (sum_local < thr):
                masks = np.delete(masks, i, axis=0)
                labels = np.delete(labels, i, axis=0)
                n_objects -= 1

        # Map from all 894 classes to 40 classes
        if self.num_labels == 40 or self.num_labels == 13:
            labels = [self.mapping40[l - 1] for l in labels]
        # Map from 40 classes to 13 classes
        if self.num_labels == 13:
            labels = [self.mapping13[l - 1] for l in labels]

        # Get bounding box coordinates for each mask.
        boxes = np.zeros((n_objects, 4))
        for i in range(n_objects):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes[i,:] = np.array([xmin, ymin, xmax, ymax])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels =  torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
