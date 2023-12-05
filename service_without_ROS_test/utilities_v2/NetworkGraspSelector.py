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

class NetworkGraspSelector:
    """
    Grasp Selector using the output of the cloth region segmentation network
    """
    def __init__(self, grasp_point_method, grasp_angle_method, grasp_target):
        self.grasp_point_method = grasp_point_method
        self.grasp_angle_method = grasp_angle_method
        self.grasp_pt = None
        self.grasp_target= grasp_target
        print("Grasp_point_method:", grasp_point_method)
        print("Grasp_angle_method:", grasp_angle_method)
        print("Grasp_target:", grasp_target)

    # not used in this project
    def sample_grasp(self, segmentation, pred):
        """Takes a 2D array prop to prob as input and sample grasp point."""
        # Filter for outer edge points w/o overlap only
        im_height, im_width, _ = segmentation.shape
        outer_edges_mask = np.zeros((im_height, im_width))
        outer_edges_mask[segmentation[:,:,1]==255] = 1

        var_map = outer_edges_mask*pred[:, :, -1]
        pvec = var_map.ravel()/np.sum(var_map) # flatten and normalize to PMF
        idx = np.random.choice(a=range(pvec.shape[0]), p=pvec)
        y,x = np.unravel_index(idx, var_map.shape)

        im = (var_map / var_map.max() * 255).astype(np.uint8)
        self.pub.publish(self.bridge.cv2_to_imgmsg(im, encoding="mono8"))

        return np.array([y,x])

    def select_grasp(self, rgb, corners, outer_edges, inner_edges, pred, retries=1, num_neighbour=8):
        impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
        impred[:, :, 0] += corners
        impred[:, :, 1] += outer_edges
        impred[:, :, 2] += inner_edges

        idxs = np.where(corners == 255)
        corners[:] = 1
        corners[idxs] = 0
        idxs = np.where(outer_edges == 255)
        outer_edges[:] = 1
        outer_edges[idxs] = 0
        idxs = np.where(inner_edges == 255)
        inner_edges[:] = 1
        inner_edges[idxs] = 0

        # Choose pixel in pred to grasp
        grasp_target = self.grasp_target
        channel = 1 if grasp_target == 'edges' else 0
        # indices由grasp_target产生,
        indices = np.where(impred[:, :, channel] == 255) # outer_edge
        if len(indices[0]) == 0:
            print("No graspable pixels detected")
            return 0, 0, 0, 0, 0

        # Choose outer point? (maybe grasp point) 
        if self.grasp_point_method == 'policy':
            outer_edges = deepcopy(pred[:, :, 1])
            mask = np.zeros_like(outer_edges)
            mask[outer_edges > 0.9] = 1
            var = deepcopy(pred[:, :, -1])
            var *= mask

            pvar = var.ravel()/np.sum(var) # flatten and normalize to PMF
            idx = np.random.choice(a=range(pvar.shape[0]), p=pvar)
            y, x = np.unravel_index(idx, var.shape)

            # 好像没有用到var_map
            var_map = (var / var.max() * 255.).astype('uint8')
            # self.pub.publish(self.bridge.cv2_to_imgmsg(var_map))
        else: 
            if self.grasp_point_method == 'manual':
                # Only works once due to rendering issues, need to restart service
                print("Manually choosing grasp point")
                wintitle = 'Choose grasp point'
                cv2.namedWindow(wintitle)
                cv2.setMouseCallback(wintitle, self.winclicked)
                cv2.imshow(wintitle, impred)
                cv2.waitKey(0)
                y, x = self.grasp_pt
            elif self.grasp_point_method == 'random':
                idx = np.random.choice(range(len(indices[0])))
                y = indices[0][idx]
                x = indices[1][idx]
            elif self.grasp_point_method == 'confidence':
                # Filter out ambiguous points
                # impred:[im_height, im_width, 3] -> corner, outer edge, inner edge predictions
                segmentation = deepcopy(impred)
                im_height, im_width, _ = segmentation.shape
                # impred的inner_edge和outer_edge同时为255,将segementation的2通道的像素置为0
                segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),2] = 0
                # impred的inner_edge和outer_edge同时为255,将segementation的1通道的像素置为0
                segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),1] = 0

                inner_edges_filt = np.ones((im_height, im_width))
                inner_edges_filt[segmentation[:,:,2]==255] = 0

                outer_edges_filt = np.ones((im_height, im_width))
                outer_edges_filt[segmentation[:,:,1]==255] = 0

                # Get outer-inner edge correspondence
                # 创建二维的坐标网格
                xx, yy =  np.meshgrid([x for x in range(im_width)],
                                    [y for y in range(im_height)])

                # Get xx_o, yy_o. (Depend on grasp_target)
                # 根据不同的grasp_target,选择对应颜色通道的像素点,保存在xx_o和yy_o
                if grasp_target == 'edges':
                    # outer_edge pixels
                    xx_o = xx[segmentation[:,:,1]==255]
                    yy_o = yy[segmentation[:,:,1]==255]
                else:
                    # corner pixels
                    xx_o = xx[segmentation[:,:,0]==255]
                    yy_o = yy[segmentation[:,:,0]==255]

                # inner_edge pixels
                xx_i = xx[segmentation[:,:,2]==255]
                yy_i = yy[segmentation[:,:,2]==255]

                # 计算内边轮廓图像离最近outer_edge像素的距离,以及每个像素的对应标签
                _, lbl = cv2.distanceTransformWithLabels(inner_edges_filt.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

                loc = np.where(inner_edges_filt==0)
                xx_inner = loc[1]
                yy_inner = loc[0]
                label_to_loc = [[0,0]]

                for j in range(len(yy_inner)):
                    label_to_loc.append([yy_inner[j],xx_inner[j]])

                label_to_loc = np.array(label_to_loc)
                direction = label_to_loc[lbl]
                print("lbl.shape:",lbl.shape)
                print("direction.shape:",direction.shape)
                # print("direction:", direction)
                # Calculate distance to the closest inner edge point for every pixel in the image
                distance = np.zeros(direction.shape)

                distance[:,:,0] = np.abs(direction[:,:,0]-yy)
                distance[:,:,1] = np.abs(direction[:,:,1]-xx)
                
                # Normalize distance vectors
                mag = np.linalg.norm([distance[:,:,0],distance[:,:,1]],axis = 0)+0.00001
                distance[:,:,0] = distance[:,:,0]/mag
                distance[:,:,1] = distance[:,:,1]/mag

                # Get distances of outer edges
                distance_o = distance[segmentation[:,:,1]==255,:]

                # Get outer edge neighbors of each outer edge point
                num_neighbour = 100

                # For every outer edge point, find its closest K neighbours 
                tree = KDTree(np.vstack([xx_o,yy_o]).T, leaf_size=2)
                dist, ind = tree.query(np.vstack([xx_o,yy_o]).T, k=num_neighbour)
                
                xx_neighbours = distance_o[ind][:,:,1]
                yy_neighbours = distance_o[ind][:,:,0]
                xx_var = np.var(xx_neighbours,axis = 1)
                yy_var = np.var(yy_neighbours,axis = 1)
                var = xx_var+yy_var
                var = (var-np.min(var))/(np.max(var)-np.min(var))
                
                # Choose min var point
                var_min = np.min(var)
                min_idxs = np.where(var == var_min)[0]
                print("Number of min var indices:", len(min_idxs))
                idx = np.random.choice(min_idxs)
                x = xx_o[idx]
                y = yy_o[idx]
            else:
                raise NotImplementedError

        print("Choose single grasp pixel x:", x, "y:", y)

        # Get outer_pt and inner_pt for computing grasp angle
        if self.grasp_angle_method == 'inneredge':
            # 计算内边缘到当前像素点的最短距离,并返回距离最近的inner_edge的像素坐标
            
            # 返回两个矩阵,temp记录每个像素点到离他最近的inner_edge的距离, lbl记录每个像素点距离最近的边缘的标记值
            temp, lbl = cv2.distanceTransformWithLabels(inner_edges.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            # 获取所有inner_edges==0的x和y值
            loc = np.where(inner_edges==0)
            xx_inner = loc[1]
            yy_inner = loc[0]

            # change
            label_to_loc = list(zip(yy_inner, xx_inner))
            # label_to_loc = zip(yy_inner, xx_inner) # AttributeError: 'zip' object has no attribute 'insert'
            label_to_loc.insert(0, (0, 0)) # 1-indexed

            label_to_loc = np.array(label_to_loc)
            direction = label_to_loc[lbl]
            outer_pt = np.array([y, x])
            inner_pt = direction[y, x]
        elif self.grasp_angle_method == 'center': # 貌似outer_pt是随机生成的,inner_pt是由物体的包围框计算,可以看作物体的几何中心
            # 随机抽取一个像素作为抓取点的位置
            # idx = np.random.choice(range(len(indices[0])))
            # y = indices[0][idx]
            # x = indices[1][idx]

            # get bbox
            bbox = deepcopy(outer_edges) if grasp_target == 'edges' else deepcopy(corners)
            
            # idxs提取掩码中非0像素点,生成一个全0的空白掩码,将idxs对应的坐标位置的像素值赋1,（将物体从边缘或者角点的形状转换成矩形）
            idxs = np.where(bbox == 0)
            bbox[:] = 0
            bbox[idxs] = 1

            # 计算bbox的尺寸和位置
            pbox = cv2.boundingRect(bbox)
            center_x = pbox[0] + 0.5*pbox[2]
            center_y = pbox[1] + 0.5*pbox[3]
            outer_pt = np.array([y, x])
            inner_pt = np.array([center_y, center_x])
        else:
            raise NotImplementedError

        # myplot(impred, xx_o, yy_o, var, outer_edges_filt, xx, yy, segmentation)
        ENABLE_PLOT = True
        if ENABLE_PLOT:
            impred2 = deepcopy(segmentation)
            impred2[:, :, 0] = 0 # 将第一个维度全部设为0

            fig = plt.figure()
            plt.figure(dpi=300)
            ax = plt.subplot(141)
            plt.title("xx_o, yy_o")
            plt.axis("off")
            empty = np.zeros(impred.shape)
            ax.imshow(empty) # 将empty显示在第一个子图中,这里呈现一个白色的矩形
            scat = ax.scatter(xx_o, yy_o, c=var, cmap='RdBu', s=3)
            # scat = ax.scatter(xx_i, yy_i, c=var, cmap='RdBu', s=3)
            # plt.colorbar(scat)
            # alpha指定散点的透明度,s指定散点的大小
            ax.scatter(x, y, c='blue', alpha=0.7, s=20)

            ax = plt.subplot(142)
            ax.imshow(impred2)
            plt.title("impred2")
            plt.axis("off")
            # plt.show()
            
            ax = plt.subplot(143)
            plt.title("outer_pt & inner_pt")
            plt.axis("off")
            empty = np.zeros(impred.shape)
            ax.imshow(empty) # 将empty显示在第一个子图中,这里呈现一个白色的矩形
            scat = ax.scatter(xx_o, yy_o, c=var, cmap='RdBu', s=3)
            # scat = ax.scatter(xx_i, yy_i, c=var, cmap='RdBu', s=3)
            # alpha指定散点的透明度,s指定散点的大小
            ax.scatter(outer_pt[1], outer_pt[0], c='blue', alpha=0.7, s=20)
            ax.scatter(inner_pt[1], inner_pt[0], c='red', alpha=0.7, s=20)
            ax.quiver(outer_pt[1], outer_pt[0], inner_pt[1] - outer_pt[1], - inner_pt[0] + outer_pt[0])

            ax = plt.subplot(144)
            plt.title("empty")
            plt.axis("off")
            ax.imshow(impred) # 将empty显示在第一个子图中,这里呈现一个白色的矩形

            # for arrow plot (not used yet, can't run)
            factor = 2
            xx = xx[outer_edges_filt==0]
            yy = yy[outer_edges_filt==0]
            direction_o = direction[segmentation[:,:,1]==255,:]

            # print("xx_o[::factor].shape:",xx_o[::factor].shape)
            # print("yy_o[::factor].shape:",yy_o[::factor].shape)
            # print("direction_o[::factor,1].shape:",direction_o[::factor,1].shape)
            # print((direction_o[::factor,1]-xx_o[::factor]).shape)
            # print((-direction_o[::factor,0]+yy_o[::factor]).shape)
            # ax.quiver(xx_o[::factor],yy_o[::factor],direction_o[::factor,1]-xx_o[::factor],-direction_o[::factor,0]+yy_o[::factor], color='white', scale=1, scale_units='x')

            base_path = "/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test/grasp_output"
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

        # count the angle
        v = inner_pt - outer_pt
        magn = np.linalg.norm(v)

        if magn == 0:
            error_msg = "magnitude is zero for %d samples" % retries
            print(error_msg)

        unitv = v / magn
        originv = [0, 1] # [y, x]
        angle = np.arccos(np.dot(unitv, originv))

        if v[0] < 0:
            angle = -angle
        
        return outer_pt[0], outer_pt[1], angle, inner_pt[0], inner_pt[1]