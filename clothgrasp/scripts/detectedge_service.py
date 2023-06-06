# -*- coding: utf-8 -*-
#!/usr/bin/env python
# Request detectedge_service, return Response(rgb_im depth_im prediction corners outer_edges inner_edges)
import os
import rospy
import cv2
import message_filters
import numpy as np
from clothgrasp.srv import DetectEdge, DetectEdgeResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyquaternion import Quaternion as Quat
from copy import deepcopy
from methods.groundtruth import GroundTruth
from methods.clothseg import ClothSegmenter
from methods.model.model import ClothEdgeModel
from methods.canny import CannyEdgeDetector
from methods.canny_color import CannyColorEdgeDetector
from methods.depthgrad import DepthGradDetector
from methods.harris import HarrisDetector
from methods.harris_color import HarrisColorDetector

class EdgeDetector():
    """
    Runs service that returns a cloth region segmentation.
    """
    def __init__(self):
        rospy.init_node('detectedge_service')
        self.bridge = CvBridge()
        self.detection_method = rospy.get_param('detection_method')
        # 初始化检测模型
        self._init_model()

        self.depth_im = None
        self.rgb_im = None
        self.depthsub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        self.rgbsub = message_filters.Subscriber('/rgb/image_raw', Image)

        self.server = rospy.Service('detect_edges', DetectEdge, self._server_cb)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.depthsub, self.rgbsub], 10, 0.1)
        self.ts.registerCallback(self._callback)

    def _init_model(self):
        if self.detection_method == 'groundtruth':
            self.crop_dims = rospy.get_param('crop_dims')
            self.model = GroundTruth(self.crop_dims)
        elif self.detection_method == 'network':
            self.crop_dims = rospy.get_param('crop_dims')
            grasp_angle_method = rospy.get_param('grasp_angle_method')
            model_path = rospy.get_param('model_angle_path') if grasp_angle_method == 'predict' else rospy.get_param('model_path')
            self.model = ClothEdgeModel(self.crop_dims, grasp_angle_method, model_path)
        elif self.detection_method == 'clothseg':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            D = np.array(rospy.get_param('D'))
            K = np.array(rospy.get_param('K'))
            K = np.reshape(K, (3, 3))
            w2c_pose = np.array(rospy.get_param('w2c_pose'))
            segment_table = rospy.get_param('segment_table')
            table_plane = np.array(rospy.get_param('table_plane'))
            self.model = ClothSegmenter(D, K, w2c_pose, segment_table, table_plane, self.crop_dims)
        elif self.detection_method == 'canny':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            self.model = CannyEdgeDetector(self.crop_dims)
        elif self.detection_method == 'canny_color':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            self.model = CannyColorEdgeDetector(self.crop_dims)
        elif self.detection_method == 'depthgrad':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            self.model = DepthGradDetector(self.crop_dims)
        elif self.detection_method == 'harris':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            self.model = HarrisDetector(self.crop_dims)
        elif self.detection_method == 'harris_color':
            self.crop_dims = rospy.get_param('crop_dims_baselines')
            self.model = HarrisColorDetector(self.crop_dims)
        else:
            raise NotImplementedError

    def _callback(self, depth_msg, rgb_msg):
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg)
        rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg)
        self.depth_im = np.nan_to_num(depth_im)
        self.rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

    def _server_cb(self, req, scale=2):
        rospy.loginfo('Received cloth detection request')

        # 深拷贝 rgb & depth image
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
            self.model.update() # Check if model needs to be reloaded
            start = rospy.Time.now()
            corners, outer_edges, inner_edges, pred = self.model.predict(depth_im)
            end = rospy.Time.now()
            d = end - start
            rospy.loginfo('Network secs: %d, nsecs: %d' % (d.secs, d.nsecs))

            response.prediction = self.bridge.cv2_to_imgmsg(pred)
            response.corners = self.bridge.cv2_to_imgmsg(corners)
            response.outer_edges = self.bridge.cv2_to_imgmsg(outer_edges)
            response.inner_edges = self.bridge.cv2_to_imgmsg(inner_edges)
        elif self.detection_method == 'clothseg':
            mask = self.model.predict(depth_im)
            response.prediction = self.bridge.cv2_to_imgmsg(mask)
        elif self.detection_method == 'canny':
            grads = self.model.predict(depth_im)
            response.prediction = self.bridge.cv2_to_imgmsg(grads)
        elif self.detection_method == 'canny_color':
            grads = self.model.predict(rgb_im)
            response.prediction = self.bridge.cv2_to_imgmsg(grads)
        elif self.detection_method == 'depthgrad':
            grads = self.model.predict(depth_im)
            response.prediction = self.bridge.cv2_to_imgmsg(grads)
        elif self.detection_method == 'harris':
            corner_preds = self.model.predict(depth_im)
            response.prediction = self.bridge.cv2_to_imgmsg(corner_preds)
        elif self.detection_method == 'harris_color':
            corner_preds = self.model.predict(rgb_im)
            response.prediction = self.bridge.cv2_to_imgmsg(corner_preds)

        rospy.loginfo('Sending cloth detection response')
        return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    e = EdgeDetector()
    e.run()
