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

def myplot(impred, xx_o, yy_o, var, outer_edges_filt, xx, yy, segmentation):
    """
    Plot for debugging
    """
    impred2 = deepcopy(segmentation)
    impred2[:, :, 0] = 0
    fig = plt.figure()
    ax = plt.subplot(121)
    empty = np.zeros(impred.shape)
    ax.imshow(empty)
    scat = ax.scatter(xx_o, yy_o, c=var, cmap='RdBu', s=3)
    plt.colorbar(scat)
    ax.scatter(x, y, c='blue', alpha=0.7)
    ax = plt.subplot(122)
    ax.imshow(impred2)
    
    # arrow
    factor = 2
    xx = xx[outer_edges_filt==0]
    yy = yy[outer_edges_filt==0]
    direction_o = direction[segmentation[:,:,1]==255,:]
    ax.quiver(xx_o[::factor],yy_o[::factor],direction_o[::factor,1]-xx_o[::factor],-direction_o[::factor,0]+yy_o[::factor], color='white', scale=1, scale_units='x')

    base_path = "/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test"
    tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    tstamp_path = os.path.join(base_path, tstamp)
    os.makedirs(tstamp_path)

    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    np.save(os.path.join(tstamp_path, "plot_%s" % tstamp), buf)
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(tstamp_path, "rgb_%s.png" % tstamp), rgb)
    plt.savefig(os.path.join(tstamp_path, 'uncertainty_%s.png' % tstamp))
    plt.show()

def classify_points(points):
    classes = []
    curr_class = []
    for i in range(len(points)):
        curr_class.append(points[i])
        if i == len(points) - 1 or ((points[i][0] == points[i+1][0] and abs(points[i][1] - points[i+1][1]) == 1) or (points[i][1] == points[i+1][1] and abs(points[i][0] - points[i+1][0]) == 1)):
            if curr_class not in classes:
                classes.append(curr_class)
            curr_class = []
        elif abs(points[i][0] - points[i+1][0]) == 1 and abs(points[i][1] - points[i+1][1]) == 1:
            curr_class.append(points[i+1])
    return classes
