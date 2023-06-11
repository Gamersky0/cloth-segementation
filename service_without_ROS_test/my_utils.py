import os
import cv2
import numpy as np
# from pyquaternion import Quaternion as Quat # seems do not need 
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from scripts.methods.groundtruth import GroundTruth # import GroundTruth OK
from scripts.methods.model.model import ClothEdgeModel # import ClothEdgeModel OK

class DetectEdgeResponse:
    # DetectEdgeResponse 包含以下成员变量
    def __init__(self):
        # numpy.ndarray
        self.rgb_im = None 
        self.depth_im = None
        self.prediction = None
        self.corners = None
        self.outer_edges = None
        self.inner_edges = None

def get_depth_img(nparray):
    depth_img = nparray.astype(np.uint16)  # 转换深度图像数据类型为uint16，便于保存
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) # 归一化操作，将深度值范围映射到[0,255]之间
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET) #利用伪彩色映射方法生成伪彩色深度图像
    return depth_colored

class SelectGraspResponse:
    def __init__(self):
        self.py = None
        self.px = None
        self.angle = None
        self.inner_py = None
        self.inner_px = None
        self.var = None
        self.angle_x = None
        self.angle_y = None