from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from image import *
from utils.other import Darknet2, load_darknet2_weights, attempt_download

import os
import sys
import time
import datetime
import argparse
import requests
import base64

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import telegram

def detect():
    single_img = False
    multi_img = False
    stream_img = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    if opt.image_url != None:
        response = requests.get(opt.image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        draw_img = np.array(Image.open(BytesIO(response.content)))
        img = transforms.ToTensor()(image)
        img, _ = pad_to_square(img, 0)
        img = resize(img, opt.img_size)
        img = img.unsqueeze(0)
        single_img = True

    elif opt.image_path != None:
        img = transforms.ToTensor()(Image.open(opt.image_path).convert("RGB"))
        draw_img = np.array(Image.open(opt.image_path))
        img, _ = pad_to_square(img, 0)
        img = resize(img, opt.img_size)
        img = img.unsqueeze(0)
        single_img = True

    elif opt.image_base64 != None:
        byte_data = base64.b64decode(opt.image)
        img = transforms.ToTensor()(Image.open(BytesIO(byte_data)).convert("RGB"))
        draw_img = np.array(Image.open(BytesIO(byte_data)))
        img, _ = pad_to_square(img, 0)
        img = resize(img, opt.img_size)
        img = img.unsqueeze(0)
        single_img = True

    elif opt.image_folder != None:
        dataloader = DataLoader(
            ImageFolder(opt.image_folder, img_size=opt.img_size),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )
        multi_img = True

    elif opt.stream != None:
        source = opt.stream
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=opt.img_size)
        stream_img = True

    # Set up model
    if(stream_img):
        model = Darknet2(opt.model_def, opt.img_size)
        attempt_download(opt.weights_path)
        if opt.weights_path.endswith(".pt"):  # pytorch format
            model.load_state_dict(torch.load(opt.weights_path, map_location=device)['model'])
        else:  # darknet format
            load_darknet2_weights(model, opt.weights_path)
        model.to(device).eval()
    else:
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        if opt.weights_path.endswith(".weights"):
        # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
        # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))

        model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if (single_img):
        single(opt, img, draw_img, model, Tensor, classes)
    elif(multi_img):
        folder(opt, dataloader, model, Tensor, classes)
    elif(stream_img):
        stream(opt, dataset, model, classes)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_base64", type=str, help="path to dataset")
    parser.add_argument("--image_path", type=str, help="path to dataset")
    parser.add_argument("--image_url", type=str, help="path to dataset")
    parser.add_argument("--image_folder", type=str, help="path to dataset")
    parser.add_argument("--stream", type=str, help="webcam or rtsp......")
    parser.add_argument("--stream_name", type=str, help="source name")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="object confidence threshold") #0.8
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")  #0.4
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
