import sys
import argparse

import cv2
import scipy.misc
from PIL import Image
import numpy as np 
from demo import resize_image, run_model_input_image


from featurize_ocr_output import load_glove, featurize_model_outputs

import warnings
warnings.filterwarnings("ignore")
f = open('codec.txt', 'r', encoding='utf-8')
codec = f.readlines()[0]
f.close()
# im = Image.fromarray(A)

def extract_images(path, frequency):
    count = 0
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    # print(success)
    success = True
    all_images = []
    t = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count))
        success,image = vidcap.read()
        cv2.imwrite("./temp/frame%d.jpg" % count, image)
        print ('Read a new frame: ', success)
        print(image.shape)
        all_images.append(image)
        count += frequency
        t += 1
        if t == 10:
            break
    return all_images


if __name__ == '__main__':
  path = './video_toy_data/maps_lyrics.mp4'
  time = 20000
  frames = extract_images(path, time)
  for f in frames:
      model_predictions = run_model_input_image(f)
      out = featurize_model_outputs(model_predictions)
      print(out)
