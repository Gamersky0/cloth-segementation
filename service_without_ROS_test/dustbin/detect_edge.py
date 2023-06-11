# -*- coding: utf-8 -*-
# Request  (none???)
# Response (rgb_im depth_im prediction corners outer_edges inner_edges)
import os
import cv2
import message_filters
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy
from scripts.methods.groundtruth import GroundTruth
from scripts.methods.model.model import ClothEdgeModel


class EdgeDetector():
    """
    Runs service that returns a cloth region segmentation.
    """
    def __init__(self):

        self.detection_method = 'network'
        # init_model
        self._init_model()
        self.depth_im = None
        self.rgb_im = None

        # 同步到 _callback 中
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depthsub, self.rgbsub], 10, 0.1)
        self.ts.registerCallback(self._callback)

    def _init_model(self):
        if self.detection_method == 'groundtruth':
            self.crop_dims = rospy.get_param('crop_dims')
            self.model = GroundTruth(self.crop_dims)
        elif self.detection_method == 'network':
            self.crop_dims = [150, 660, 415, 900, 2]
            grasp_angle_method = 'inneredge'
            
            # grasp_angle_method 表示程序在计算抓取角度时要使用的方法
            # 如果 grasp_angle_method 为 'predict'，说明程序要使用的模型是用于预测抓取角度的模型，
            # 因此 model_path 参数将被替换为 model_angle_path 参数，即程序将使用不同的模型。
            # 否则，model_path 参数保持不变，程序将使用默认的模型。
            # 因此，这行代码的作用是根据 grasp_angle_method 的值获取正确的模型路径，并将其保存在 model_path 变量中。
            
            model_path = "/home/chimy/old_projects/cloth-segmentation-main/runspath/pretrained_weights"
            self.model = ClothEdgeModel(self.crop_dims, grasp_angle_method, model_path)
        else:
            raise NotImplementedError

    def run(self):
        # 原本是 callback 的内容
        # set self.depth_im & self.rgb_im， 在这里取固定的depth和image
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg)
        rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg)

        rgb = Image.open(os.path.join(self.datapath, "rgb_%d.png" % i))
        depth = np.load(os.path.join(self.datapath, "%d_depth.npy" % i))

        # N: 将 self.depth_im 中的 NaN 值替换为 0。防止在图像处理中出现 NaN 导致错误
        self.depth_im = np.nan_to_num(depth_im)
        # N: 将输入的 BGR 图像转换为 RGB 图像
        self.rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

        # _server_cb content
        print("Received cloth detection request")

        # deepcopy rgb & depth image
        rgb_im = deepcopy(self.rgb_im)
        depth_im = deepcopy(self.depth_im)
        if rgb_im is None or depth_im is None:
            raise rospy.ServiceException('Missing RGB or Depth Image')

        # 这段代码是用于构造 ROS 服务（Service）的返回消息。
        # 创建了一个名为 DetectEdgeResponse 的服务返回消息，然后将处理后的 rgb_im 和 depth_im 图像转换为 ROS 消息（Message），
        # 并设置到 DetectEdgeResponse 消息中的 rgb_im 和 depth_im 属性中。

        # # DetectEdgeResponse 包含以下成员变量
        # sensor_msgs/Image rgb_im
        # sensor_msgs/Image depth_im
        # sensor_msgs/Image prediction
        # sensor_msgs/Image corners
        # sensor_msgs/Image outer_edges
        # sensor_msgs/Image inner_edges
        response = DetectEdgeResponse()
        response.rgb_im = self.bridge.cv2_to_imgmsg(rgb_im)
        response.depth_im = self.bridge.cv2_to_imgmsg(depth_im)

        if self.detection_method == 'groundtruth':
            pred = self.model.predict(rgb_im)
            response.prediction = self.bridge.cv2_to_imgmsg(pred)
        elif self.detection_method == 'network':
            # self.model.update() # Check if model needs to be reloaded
            # start = rospy.Time.now()
            corners, outer_edges, inner_edges, pred = self.model.predict(depth_im)
            # end = rospy.Time.now()
            # d = end - start
            # print("Network secs:", d.secs, "nsecs: ", d.nsecs)

            response.prediction = pred
            response.corners = corners
            response.outer_edges = outer_edges
            response.inner_edges = inner_edges
        
        print("Sending cloth detection response")
        return response


if __name__ == '__main__':
    e = EdgeDetector()
    e.run()
