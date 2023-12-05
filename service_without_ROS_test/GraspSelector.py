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


class GraspSelector_deprecated:
    def __init__(self, detection_method, grasp_point_method, grasp_angle_method, img_prediction, grasp_target):
        self.detection_method = detection_method
        self.grasp_point_method = grasp_point_method
        self.grasp_angle_method = grasp_angle_method
        self.img_prediction = img_prediction
        self.grasp_target = grasp_target
        self._init_selector()

    def _init_selector(self):
        # self.selector = XXX
        if self.detection_method == 'groundtruth':
            print("Not define GroundTruthSelector")
            # self.selector = GroundTruthSelector()
        elif self.detection_method == 'network':
            self.selector = NetworkGraspSelector(self.grasp_point_method, self.grasp_angle_method, self.grasp_target)
        else:
            raise NotImplementedError
        print("Success init selector.")
        return
    
    def run(self):
        img_prediction = self.img_prediction
        inner_py = None
        inner_px = None

        if self.detection_method == 'groundtruth':
            pred = deepcopy(img_prediction.prediction)
            py, px, angle, inner_py, inner_px, var, angle_x, angle_y = self.selector.select_grasp(pred)
        elif self.detection_method == 'network':
            corners = deepcopy(img_prediction.corners)
            outer_edges = deepcopy(img_prediction.outer_edges)
            inner_edges = deepcopy(img_prediction.inner_edges)
            rgb = deepcopy(img_prediction.rgb_im)
            pred = deepcopy(img_prediction.prediction)
            py, px, angle, inner_py, inner_px = self.selector.select_grasp(rgb, corners, outer_edges, inner_edges, pred)
        else:
            prediction = deepcopy(img_prediction.prediction)
            py, px, angle = self.selector.select_grasp(prediction)
            print(py, px, angle)

        response = SelectGraspResponse()
        response.py = py
        response.px = px
        response.angle = angle
        # print inner_px & inner_py
        if inner_py != None and inner_px != None:
            response.inner_py = inner_py
            response.inner_px = inner_px
        if self.detection_method == 'groundtruth' and px != 0:
            response.var = var.flatten()
            response.angle_x = angle_x.flatten()
            response.angle_y = angle_y.flatten()

        print("Get grasp selection response:")
        print('{:<10} {:<5} {:<10} {:<5}'.format("px:", px, "py:", py))
        print('\033[32m' + '{:<10} {} '.format("angle:", angle) + '\033[0m')
        print('{:<10} {:<5} {:<10} {:<5}'.format("inner_px:", inner_px, "inner_py:", inner_py))

        # self.selector.plot(pred, px, py, var, None, inner_px, inner_py, prediction.prediction)
        return response