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

class EdgeDetector:
    def __init__(self, detection_method, crop_dims, datapath):
        self.detection_method = detection_method
        self.datapath = datapath
        self._init_model(crop_dims)
        self.depth_im = None
        self.rgb_im = None
        print("Finish EdgeDetector init.")

    def _init_model(self, crop_dims):
        if self.detection_method == 'groundtruth':
            self.crop_dims = crop_dims
            self.model = GroundTruth(self.crop_dims)
        elif self.detection_method == 'network':
            self.crop_dims = crop_dims
            grasp_angle_method = 'inneredge'
            
            # grasp_angle_method 表示程序在计算抓取角度时要使用的方法
            # 如果 grasp_angle_method 为 'predict'，说明程序要使用的模型是用于预测抓取角度的模型，
            # 因此 model_path 参数将被替换为 model_angle_path 参数，即程序将使用不同的模型。
            # 否则，model_path 参数保持不变，程序将使用默认的模型。
            # 因此，这行代码的作用是根据 grasp_angle_method 的值获取正确的模型路径，并将其保存在 model_path 变量中。
            
            model_path = "/home/chimy/old_projects/cloth-segmentation-main/runspath/pretrained_weights"
            self.model = ClothEdgeModel(self.crop_dims, grasp_angle_method, model_path)

    def detect_edge(self, i):
        try:
            # Self Image Test (Fail)
            # rgb_im = Image.open("/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test/my_image/my_rgb_1.png")
            # depth_im = np.load("/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test/my_image/my_dep_1_normal.npy")
            rgb_im = Image.open(os.path.join(self.datapath, "rgb_%d.png" % i))
            depth_im = np.load(os.path.join(self.datapath, "%d_depth.npy" % i))
            max_d = np.nanmax(depth_im)
            depth_im[np.isnan(depth_im)] = max_d
            print("Input depth_image path:", os.path.join(self.datapath, "%d_depth.npy" % i))
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to read file")
        
        # Prevents NaN from causing errors in image processing
        self.depth_im = np.nan_to_num(depth_im)
        self.rgb_im = cv2.imread(os.path.join(self.datapath, "rgb_%d.png" % i))

        # _server_cb content
        print("Received cloth detection request")

        rgb_im = deepcopy(self.rgb_im)  
        depth_im = deepcopy(self.depth_im)

        response = DetectEdgeResponse()
        response.rgb_im = self.rgb_im
        response.depth_im = self.depth_im
        # get response and save images
        if self.detection_method == 'groundtruth':
            pred = self.model.predict(rgb_im)
            # pred = self.model.predict(rgb_im)
            response.prediction = pred
        elif self.detection_method == 'network':
            print("Detectopm method: network")
            self.model.update() # Check if model needs to be reloaded
            print("Input depth_image.shape: ", depth_im.shape)
            # all numpy.ndarray
            corners, outer_edges, inner_edges, pred = self.model.predict(depth_im)
            print("Cost time: not add yet")

            # save prediction 
            corners_img = get_depth_img(corners)
            cv2.imwrite('corners_img.png', corners_img)
            outer_img = get_depth_img(outer_edges)
            cv2.imwrite('outer_img.png', outer_img)
            inner_img = get_depth_img(inner_edges)
            cv2.imwrite('inner_img.png', inner_img)
            pred_img = get_depth_img(pred)
            cv2.imwrite('pred_img.png', pred_img)

        ENBALE_DETECT_EDGE_PLOT = True
        if ENBALE_DETECT_EDGE_PLOT:
            plt.figure(dpi=300)
            plt.subplot(141)
            plt.title("depth_image")
            plt.imshow(depth_im)
            plt.axis("off")

            plt.subplot(142)
            plt.title("corners")
            plt.imshow(corners)
            plt.axis("off")

            plt.subplot(143)
            plt.title("outer_edges")
            plt.imshow(outer_edges)
            plt.axis("off")

            plt.subplot(144)
            plt.title("inner_edges")
            plt.imshow(inner_edges)
            plt.axis("off")
            plt.show()

        response.prediction = pred
        response.corners = corners
        response.outer_edges = outer_edges
        response.inner_edges = inner_edges
        # TYPE_TEST = True
        # if TYPE_TEST:
        #     print("response.prediction.type:",response.prediction.type)
        #     print("response.corners.type:",response.corners.type)
        #     print("response.outer_edges.type:",response.outer_edges.type)
        #     print("response.inner_edges.type:",response.inner_edges.type)
        return response

    def run(self, img_index):
        # set self.depth_im & self.rgb_im, 
        # img_index: dataset image index
        index = img_index

        # DetectEdgeResponse
        prediction = self.detect_edge(index)
        
        return prediction