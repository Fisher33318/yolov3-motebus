from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import requests
import base64
import datetime
import json
import cv2
import subprocess

from PIL import Image

from utils.other import scale_coords, plot_one_box


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

bot = telegram.Bot('641293137:AAElcXvCTIPA7vHsTOiiallG9gt7PSfPRSo')
chat_id=-347773088
ttt = time.time()
tttt = time.time()
mid=[]
bsize=[]
left = 120  #+-20
right = 500

rtmp = 'rtmp://localhost/live/test'
command = ['ffmpeg',
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', "640x480",
    '-r', '25',
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'flv',
    rtmp]

pipe = subprocess.Popen(command
    , shell=False
    , stdin=subprocess.PIPE
    , stdout=subprocess.DEVNULL   #hide print
    , stderr=subprocess.STDOUT   #hide print
)


def single(opt, img, draw_img, model, Tensor, classes):
    prev_time = time.time()
    input_img = Variable(img.type(Tensor))
    detections = model(input_img)
    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    current_time = time.time()
    inference_time = current_time - prev_time
    print("+Inference Time: %s" % (inference_time))
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]   
    draw_img = draw_img
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(draw_img)
    # Draw bounding boxes and labels of detections
    detections = detections[0]
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, draw_img.shape[:2])
        unique_labels = detections[:, -1].unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            box_w = x2 - x1
            box_h = y2 - y1
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]##
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = "result"
    plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    bot.sendPhoto(chat_id=-347773088, photo=open("output/%s.png" % filename, "rb"))
    plt.close()


def folder(opt, dataloader, model, Tensor, classes):
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    prev_time = time.time()

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        #bot = telegram.Bot('641293137:AAElcXvCTIPA7vHsTOiiallG9gt7PSfPRSo')
        #chat_id=-347773088
        #bot.sendPhoto(chat_id=-347773088, photo=open("output/%s.png" % filename, "rb"))
        plt.close()

def stream(opt, dataset, model,  classes):
    global ttt, tttt, mid, bsize
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(0)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        pred = model(img)[0]
        t2 = time.time()


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        det = pred[0]

        p, s, im0 = path[0],  '', im0s[0]
#        s += '%gx%g ' % img.shape[2:]  # print string
        tt = time.time()
        if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
            for *xyxy, conf, cls in det:
                if (conf > 0.96):
                  label = '%s %.2f' % (classes[int(cls)], conf)
                  plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                  if (classes[int(cls)] == "person" and abs(tt- tttt) > 0.2):
                      mid.append(((int(xyxy[0]) + int(xyxy[2]))/2, (int(xyxy[1]) + int(xyxy[3]))/2))
                      bsize.append(abs(int(xyxy[0]) - int(xyxy[2])) + abs(int(xyxy[1]) - int(xyxy[3])))
                      tttt = time.time()
                  elif (abs(tt- tttt) > 3):
                      mid = []
                      bsize = []
        if (len(mid) > 13): 
            '''
            if(mid[0][0] - mid[-1][0] > 0 and mid[0][0] >= right):                   #R - L     wel
                mes = "Welcome to %s !" % opt.stream_name
                mid = []
                bot.sendMessage(chat_id, mes)
            elif(mid[0][0] - mid[-1][0] > 0 and left < mid[0][0] < right):            #R - L     bye
                mes = "Come again!"
                mid = []
                bot.sendMessage(chat_id, mes)
            elif(mid[0][0] - mid[-1][0] < 0 and mid[0][0] <= left):                 #L - R     wel
                mes = "Welcome to %s !" % opt.stream_name
                mid = []
                bot.sendMessage(chat_id, mes)
            elif(mid[0][0] - mid[-1][0] < 0 and right > mid[0][0] > left):              #L - R     bye
                mes = "Come again!"
                mid = []
                bot.sendMessage(chat_id, mes)
            '''
            print(mid[-1][0])
            if(bsize[0] - bsize[-1] < 0 and  left <= mid[-1][0] <= right):                   #in L - R  small to large
                mes = "Welcome to %s !" % opt.stream_name
                mid = []
                bsize = []
                #bot.sendMessage(chat_id, mes)
            elif(bsize[0] - bsize[-1] > 0 and left <= mid[-1][0] <= right):            #in L - R  large to small
                mes = "Come again!"
                mid = []
                bsize = []
                #bot.sendMessage(chat_id, mes)
            if("person" in s):
                if (abs(tt - ttt) > 8):
                    cv2.imwrite("output/stream.png", im0)
                    dt = datetime.datetime.now()
                    mes = "%s: %s from %s" % (s, dt.strftime('%Y-%m-%d %H:%M:%S'), opt.stream_name)
                    ttt = time.time()
                    #bot.sendMessage(chat_id, mes)
                    #bot.sendPhoto(chat_id, photo=open("output/stream.png", "rb"))


            # Print time (inference + NMS)
        #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
        cv2.imshow("screen", im0)
        pipe.stdin.write(im0.tostring())
        with open('cmd.json', 'r') as reader:
            cmd = json.loads(reader.read())
        if cmd['stream_name'] == opt.stream_name and cmd['method'] == "stop":
            data_json = {"stream_name": "", "method": 'stop'}
            with open('cmd.json', 'w') as f:
                json.dump(data_json, f)
            break




    #print('Done. (%.3fs)' % (time.time() - t0))
