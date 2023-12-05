import os
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from scripts.methods.groundtruth import GroundTruth # import GroundTruth OK
from scripts.methods.model.model import ClothEdgeModel # import ClothEdgeModel OK
from my_utils import *
from datetime import datetime
from sklearn.neighbors import KDTree
from NetworkGraspSelector import *
from EdgeDetector import *
from GraspSelector import *

def detect_edge():
    detection_method = 'network'
    datapath = "/home/chimy/projects/biyesheji/data_painted_towel"
    crop_dims = [180, 650, 450, 900, 2] # seems not used yet
    img_index = 1

    e = EdgeDetector(detection_method, crop_dims, datapath)
    prediction = e.run(img_index)
    return prediction

def select_grasp(prediction):
    # Option: network
    detection_method = "network"
    # Option: random, manual, confidence, policy
    grasp_point_method = "confidence"
    # Option: predict没有写?, inneredge, center
    grasp_angle_method = "center"
    # Option: corner, edges
    grasp_target = 'corner'

    g = GraspSelector(detection_method, grasp_point_method, grasp_angle_method, prediction, grasp_target)
    grasp = g.run()
    return grasp

if __name__ == '__main__':
    print('\033[32m' + '==========================================' + '\033[0m')
    print('\033[32m' + 'Begin detect_edge process' + '\033[0m')
    prediction = detect_edge()
    print('\033[32m' + 'End detect_edge process' + '\033[0m')
    print('\033[32m' + '==========================================' + '\033[0m')

    # print prediction shape 
    ENBALE_PREDICTION_SHAPE_TEST = True
    if ENBALE_PREDICTION_SHAPE_TEST:
        print('{:<25} {}'.format("prediction.rgb_im.shape:", prediction.rgb_im.shape))
        print('{:<25} {}'.format("prediction.depth_im.shape:", prediction.depth_im.shape))
        print('{:<25} {}'.format("prediction.prediction.shape:", prediction.prediction.shape))
        print('{:<25} {}'.format("prediction.corners.shape:", prediction.corners.shape))
        print('{:<25} {}'.format("prediction.outer_edges.shape:", prediction.outer_edges.shape))
        print('{:<25} {}'.format("prediction.inner_edges.shape:", prediction.inner_edges.shape))

    print('\033[32m' + '==========================================' + '\033[0m')
    print('\033[32m' + 'Begin select_grasp process' + '\033[0m')
    grasp = select_grasp(prediction)
    print('\033[32m' + 'End select_grasp process' + '\033[0m')
    print('\033[32m' + '==========================================' + '\033[0m')