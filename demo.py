'''
Created on Aug 25, 2017

@author: busta
'''

import cv2
import numpy as np
import os
import warnings

from nms import get_boxes

from models import ModelResNetSep2
import net_utils

from ocr_utils import ocr_image
from data_gen import draw_box_points
import torch

import argparse

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import scipy.misc

f = open('codec.txt', 'r', encoding='utf-8')
codec = f.readlines()[0]
f.close()

def resize_image(im, max_size = 1585152, scale_up=True):

  if scale_up:
    image_size = [im.shape[1] * 3 // 32 * 32, im.shape[0] * 3 // 32 * 32]
  else:
    image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
  while image_size[0] * image_size[1] > max_size:
    image_size[0] /= 1.2
    image_size[1] /= 1.2
    image_size[0] = int(image_size[0] // 32) * 32
    image_size[1] = int(image_size[1] // 32) * 32


  resize_h = int(image_size[1])
  resize_w = int(image_size[0])


  scaled = cv2.resize(im, dsize=(resize_w, resize_h))
  return scaled, (resize_h, resize_w)

def valid_image_size(image_path, width):
    number_component = image_path.split("_")[1].split(".")[0]
    num = int(number_component)
    return num == width

def valid_image_number(image_path, img_num):
  #indexings: [-2:], [-1]
  if img_num == -1:
    return True
  try:
      number_component = image_path.split("_")[-2][-2:]
      num = int(number_component)
      return num == img_num
  except:
    # print("invalid")
    return False

def run_model_input_image(im, show_boxes=False):
  predictions = {}
  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='e2e-mlt-rctw.h5')
  parser.add_argument('-segm_thresh', default=0.5)

  font2 = ImageFont.truetype("Arial-Unicode-Regular.ttf", 18)

  args = parser.parse_args()

  net = ModelResNetSep2(attention=True)
  net_utils.load_net(args.model, net)
  net = net.eval()

  if args.cuda:
    print('Using cuda ...')
    net = net.cuda()

  with torch.no_grad():
    # im = Image.open(im)
    # im = im.convert('RGB')
    im = np.asarray(im)
    im = im[...,:3]
    im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
    images = np.asarray([im_resized], dtype=np.float)
    images /= 128
    images -= 1
    im_data = net_utils.np_to_variable(images, is_cuda=args.cuda).permute(0, 3, 1, 2)
    seg_pred, rboxs, angle_pred, features = net(im_data)

    rbox = rboxs[0].data.cpu()[0].numpy()
    rbox = rbox.swapaxes(0, 1)
    rbox = rbox.swapaxes(1, 2)

    angle_pred = angle_pred[0].data.cpu()[0].numpy()


    segm = seg_pred[0].data.cpu()[0].numpy()
    segm = segm.squeeze(0)

    draw2 = np.copy(im_resized)
    boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

    img = Image.fromarray(draw2)
    draw = ImageDraw.Draw(img)

    #if len(boxes) > 10:
    #  boxes = boxes[0:10]

    out_boxes = []
    prediction_i = []
    for box in boxes:

        pts  = box[0:8]
        pts = pts.reshape(4, -1)

        det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
        if len(det_text) == 0:
            continue

        width, height = draw.textsize(det_text, font=font2)
        center =  [box[0], box[1]]
        draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
        out_boxes.append(box)

        # det_text is one prediction
        prediction_i.append(det_text.lower())

    predictions["frame"] = prediction_i

    # show each image boxes and output in pop up window.
    show_image_with_boxes(img, out_boxes, show=show_boxes)

  print(predictions)
  return predictions

def run_model(path, show_boxes=False):
  predictions = {}
  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='e2e-mlt-rctw.h5')
  parser.add_argument('-segm_thresh', default=0.5)

  font2 = ImageFont.truetype("Arial-Unicode-Regular.ttf", 18)

  args = parser.parse_args()

  net = ModelResNetSep2(attention=True)
  net_utils.load_net(args.model, net)
  net = net.eval()

  if args.cuda:
    print('Using cuda ...')
    net = net.cuda()

  image = os.listdir(path)
  frame_no = 0
  with torch.no_grad():

    for i in image:
      if valid_image_number(path+i, 15):
        im=Image.open(path + i)

        im = im.convert('RGB')
        im = np.asarray(im)
        im = im[...,:3]
        im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
        images = np.asarray([im_resized], dtype=np.float)
        images /= 128
        images -= 1
        im_data = net_utils.np_to_variable(images, is_cuda=args.cuda).permute(0, 3, 1, 2)
        seg_pred, rboxs, angle_pred, features = net(im_data)

        rbox = rboxs[0].data.cpu()[0].numpy()
        rbox = rbox.swapaxes(0, 1)
        rbox = rbox.swapaxes(1, 2)

        angle_pred = angle_pred[0].data.cpu()[0].numpy()


        segm = seg_pred[0].data.cpu()[0].numpy()
        segm = segm.squeeze(0)

        draw2 = np.copy(im_resized)
        boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

        img = Image.fromarray(draw2)
        draw = ImageDraw.Draw(img)

        #if len(boxes) > 10:
        #  boxes = boxes[0:10]

        out_boxes = []
        prediction_i = []
        for box in boxes:

          pts  = box[0:8]
          pts = pts.reshape(4, -1)

          det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
          if len(det_text) == 0:
            continue

          width, height = draw.textsize(det_text, font=font2)
          center =  [box[0], box[1]]
          draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
          out_boxes.append(box)

          # det_text is one prediction
          prediction_i.append(det_text)

        predictions[i] = prediction_i

        # show each image boxes and output in pop up window.
        show_image_with_boxes(img, out_boxes, show=show_boxes)

  print(predictions)
  return predictions

def show_image_with_boxes(img, out_boxes, show=False):
  if show:
    im = np.array(img)
    for box in out_boxes:
      pts  = box[0:8]
      pts = pts.reshape(4, -1)
      draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

    cv2.imshow('img', im)
    cv2.waitKey(1)


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  path = './img_quality_experiments_data/'
  images = "all"
  run_model(path, show_boxes = False)
