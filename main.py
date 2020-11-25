import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import time

import helperFunctions as helper

image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
dowloaded_image_path = helper.download_and_resize_image(image_url, 1280, 856, True)
print("done")

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" 

detector = hub.load(module_handle).signatures['default']

print("done detector")

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, path):
    img = load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    print("add boxes")
    image_with_boxes = helper.draw_boxes(img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])
    print("display image")
    helper.display_image(image_with_boxes)
    print("image should display")
    time.sleep(60)

run_detector(detector, dowloaded_image_path)