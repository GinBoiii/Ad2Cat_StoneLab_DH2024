"""Attempt at running this on something"""

# Imports
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os

import pandas as pd
import numpy as np

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image


# YoloV1
architecture_config = [
    # Tuple: (kernel_size, number of filters, strides, padding)
    # "M" = Max Pool Layer
    # List: [(tuple), (tuple), how many times to repeat]
    # Doesnt include fc layers
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))


class YoloV1(nn.Module):
    def __init__(self):
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = 3
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]  # Tuple
                conv2 = x[1]  # Tuple
                repeats = x[2]  # Int

                for _ in range(repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self):
        B = 2
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
class YoloDataset2(torch.utils.data.Dataset):
    def __init__(self, img_list, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = img_list

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
            img_path = self.images[index]
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = 7
        self.B = 2
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        c = self.C
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)
            boxes = torch.tensor(boxes)

            if self.transform:
                image, boxes = self.transform(image, boxes)

            label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
            for box in boxes:
                class_label, x, y, width, height = box.tolist()
                class_label = int(class_label)
                i, j = int(self.S * y), int(self.S * x)
                x_cell, y_cell = self.S * x - j, self.S * y - i
                width_cell, height_cell = (width * self.S, height * self.S)
                if label_matrix[i, j, c] == 0:
                    label_matrix[i, j, c] = 1
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    label_matrix[i, j, (c + 1) : (c + 5)] = box_coordinates
                    label_matrix[i, j, class_label] = 1

            return image, label_matrix


def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4) (x, y, w, h)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4) (x, y, w, h)

    Returns:
        tensor: Intersection over union for all examples (BATCH_SIZE, 1) (box_iou)
    """

    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they don't intersect. Since when they don't intersect, one of these will be negative so that should become 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bbox
        specified as [class_pred, prob_score, x, y, w, h]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    threshold = THRESHOLD
    iou_threshold = IOU_THRESHOLD

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_bboxes(loader, model):

    iou_threshold = IOU_THRESHOLD
    threshold = THRESHOLD
    device = DEVICE

    all_pred_boxes = []
    all_true_boxes = []
    all_image_indices = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for _, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx])

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
                all_image_indices.append(train_idx + 1)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes, all_image_indices


def get_bboxes2(input, model):
    
    iou_threshold = IOU_THRESHOLD
    threshold = THRESHOLD
    device = DEVICE

    all_pred_boxes = []
    all_true_boxes = []
    all_image_indices = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    x = input.to(device)

    with torch.no_grad():
        predictions = model(x)

    bboxes = cellboxes_to_boxes(predictions)

    nms_boxes = non_max_suppression(bboxes[0])

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

    for nms_box in nms_boxes:
        if nms_box[1] > threshold:
            all_pred_boxes.append([train_idx] + nms_box)
            all_image_indices.append(train_idx + 1)


    train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes, all_image_indices


def cellboxes_to_boxes(out):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def convert_cellboxes(predictions):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 10)
    bboxes1 = predictions[..., C + 1 : C + 5]
    bboxes2 = predictions[..., C + 6 : C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def plot_image(image, boxes, box_indices):
    """Plots predicted bounding boxes on the image"""

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = np.array(image)
    height, width = im.shape

    # Create figure and axes
    _, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for i in range(len(box_indices)):
        # box = boxes[i][2:-1]
        box = boxes[i][3:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


# First lets define some Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

PIN_MEMORY = True

CLASSES = ["Add"]
C = len(CLASSES)
S = 7

EXPERIMENT_DIR = "/experiment"

IMAGE_DIR = "images"
LABEL_DIR = "labels"


def normal_transform(image, bboxes):
    transform = transforms.Compose(
        [transforms.Resize((416, 416)), transforms.PILToTensor()]
    )
    image = transform(image).float()
    return image, bboxes

def unormal_transform(image):
    transform = transforms.Compose(
        [transforms.Resize((416, 416)), transforms.PILToTensor()]
    )
    image = transform(image).float()
    return image


# Secondly define model and load it from .pth file
model = YoloV1().to(DEVICE)
# Load the model
state_dict = torch.load("trained_model.pth", map_location=torch.device("cpu"))
# print(state_dict.keys())
model.load_state_dict(state_dict["state_dict"])
"""
model_path = "trained_model.pth"
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    # If CUDA is not available, load the model on the CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
"""
transform = normal_transform

experiment_dataset = YoloDataset(
    "experiment.csv",
    transform=transform,
    img_dir=IMAGE_DIR + EXPERIMENT_DIR,
    label_dir=LABEL_DIR + EXPERIMENT_DIR,
)

experiment_loader = DataLoader(
    dataset=experiment_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True,
)

pred_boxes, true_boxes, box_indices = get_bboxes(experiment_loader, model)

print("||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||||||||||||||||||||||||")
print("Predicted boxes: ", pred_boxes)
print("Box indexes", box_indices)
print("|||||||||||||||||||||get_metrics|||||||||||||||||||||||")
print("Printing images . . .")


#for i in range(1):
#    plot_image(f"images/experiment/{i+1}_add.jpg", pred_boxes, box_indices)


def overlay_image(image, overlay_image_path, boxes, box_indices):
    """Overlays an image on the original image based on the bounding boxes"""

    # Read the original image
    original_image = torch.squeeze(image)
    #print(original_image.shape)
    #print('l')
    original_image = original_image.permute(1, 2, 0)
    #print(original_image.shape)

    # Read the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Loop through bounding boxes and overlay the image
    for i in range(len(box_indices)):
        box = boxes[i][3:]
        print(box)
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = int((box[0] - box[2] / 2) * original_image.shape[1])
        upper_left_y = int((box[1] - box[3] / 2) * original_image.shape[0])
        bottom_right_x = int((box[0] + box[2] / 2) * original_image.shape[1])
        bottom_right_y = int((box[1] + box[3] / 2) * original_image.shape[0])

        # Resize the overlay image to fit the bounding box
        overlay_resized = cv2.resize(
            overlay_image,
            (bottom_right_x - upper_left_x, bottom_right_y - upper_left_y),
        )

        # Overlay the image
        overlay_resized = torch.tensor(overlay_resized)
        print('d')
        print(overlay_resized.shape)
        original_image[upper_left_y:bottom_right_y, upper_left_x:bottom_right_x] = (
            overlay_resized
        )

    # Display the overlaid image
    return original_image
    #cv2.destroyAllWindows() 


# Call overlay_image function with appropriate image and overlay image paths
'''
overlay_image(
    "images/experiment/1_add.jpg",  # Original image path
    "images/overlay/overlay_image.jpg",  # Overlay image path
    pred_boxes,  # Predicted boxes
    box_indices,  # Box indices
)
'''
# Initialize camera
cap = cv2.VideoCapture(0)  # Use default camera (change to the appropriate device index if needed)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    pil_image = Image.fromarray(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = unormal_transform(pil_image)
    frame = torch.unsqueeze(frame, dim = 0)
    pred_boxes, true_boxes, box_indices = get_bboxes2(frame, model)

    frame = overlay_image(
        frame,  # Original image path
        "images/overlay/overlay_image.jpg",  # Overlay image path
        pred_boxes,  # Predicted boxes
        box_indices,  # Box indices
    )

    print(pred_boxes)

    if not ret:
        break
    
    
    
    # Display the frame
    print(frame.shape)
    frame = frame.numpy()
    frame = (frame).astype(np.uint8) 
    cv2.imshow('Frame', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

