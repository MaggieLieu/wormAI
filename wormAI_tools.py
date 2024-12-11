import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np 
import matplotlib.pyplot as plt
import random
import cv2
from scipy.interpolate import interp1d
import matplotlib
import keras
from keras import ops
import math
import keras_cv

def split_value_into_segments(value, n):
    """Splits a value into n equal segments and returns the segment values."""
    segment_size = int(value / n)
    segments = [segment_size * i for i in range(n+1)]  # Include endpoints
    return segments

def get_length(x,y):
    "Returns length of worm given a set of coordinates" 
    worm_length = 0
    for i in range(len(x)-1):
        worm_length += np.sqrt( (x[i+1]-x[i])**2 + (y[i+1] - y[i])**2)
    return worm_length

def get_worm(w=300, h=300, length=250, wiggle=50, r=10, ns=5):
    "Simulate a worm cropped to fit image size"
    #w: box width default 300
    #h: box height default 300
    #length: length of worm
    #wiggle: amount of wiggle default 50
    #r: radius of worm default 10

    xs = np.array(split_value_into_segments(length, ns)) #assumes startpoint at (0,0) and endpt at (l,?)
    ys = np.hstack([0, np.random.standard_t(10, ns)*wiggle]) #,np.random.randn(ns)*wiggle]) # y values of center

    ysp = ys + r*0.5
    ysm = ys - r*0.5

    ysp[0] = ys[0]
    ysp[-1]= ys[-1]
    ysm[0] = ys[0]
    ysm[-1]= ys[-1]

    spline = interp1d(xs, ys, kind='cubic')
    splinep =  interp1d(xs, ysp, kind='cubic')
    splinem =  interp1d(xs, ysm, kind='cubic')

    # Sample points from the spline curve
    sample_x = np.arange(min(xs), max(xs))
    sample_y = spline(sample_x).astype(int)
    sample_ysp = splinep(sample_x).astype(int)
    sample_ysm = splinem(sample_x).astype(int)

    ymax = np.max(sample_ysp)
    ymin = np.min(sample_ysm)
    xmax = np.max(xs)
    xmin = np.min(xs)
    dy = ymax - ymin
    dx = xmax - xmin

    sim_im = np.zeros((dy,dx))
    norm_ysm = sample_ysm+np.abs(ymin)
    norm_ysp = sample_ysp+np.abs(ymin)

    for i in sample_x:
        sim_im[norm_ysm[i]:norm_ysp[i], i] = 1

    worm_length = int(get_length(sample_x, sample_y))

    return worm_length, sim_im

def generate_worm_sim(w=500, h=500, length=250, wiggle=(250//10),r=10,ns=5):

    worm_length, sim_worm = get_worm(w,h,length, wiggle, r, ns) 


    "Get Worm Simulation"
    # Randomly choose a rotation angle
    angle = np.random.uniform(0, 360)

    #Get im dimensions
    rows, cols = sim_worm.shape[:2]

    #calculate rotation matrix
    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1.0)

    # Rotate worm
    rotated_worm = cv2.warpAffine(sim_worm, M, (cols, rows))


    bh, bw = np.shape(rotated_worm)
    pad_top = random.randint(0, 1024 - bh)
    pad_left = random.randint(0, 1360 - bw)
    pad_bottom = 1024 - pad_top - bh
    pad_right =  1360 - pad_left - bw

    full_sim = np.pad(rotated_worm, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    full_sim = cv2.resize(full_sim, (256,256)) #(y,x)

    scalex = 256/1360
    scaley = 256/1024
    x = int((pad_left + bw/2) * scalex)
    y = int((pad_top + bh/2) * scaley)
    dx = int(bw * scalex)
    dy = int(bh * scaley)
    return full_sim, x,y, dx, dy

def get_detection_model():
    tf.keras.backend.clear_session()
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
    od_model = keras_cv.models.YOLOV8Detector(
                num_classes=1,
                    bounding_box_format="center_xywh",
                        backbone=backbone,
                            fpn_depth=1,)
    od_model.load_weights('localisation_model.keras') #load model
    return od_model
