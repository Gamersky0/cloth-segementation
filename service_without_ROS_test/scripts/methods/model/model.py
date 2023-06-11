# -*- coding: utf-8 -*-
#!/usr/env/bin python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
from scripts.methods.model.unet import unet
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as T
from PIL import Image
import cv2

class ClothEdgeModel:
    def __init__(self, crop_dims, predict_angle, model_path):
        self.predict_angle = predict_angle
        self.model_path = model_path
        self.model_last_updated = os.path.getmtime(self.model_path)
        self.use_gpu = torch.cuda.is_available()
        self.num_gpu = list(range(torch.cuda.device_count()))
        self.crop_dims = crop_dims

        self.transform = T.Compose([T.ToTensor()])

        self.n_class = 3
        # self.n_class = 6
        self.model = unet(n_classes=self.n_class, in_channels=1)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        if self.use_gpu:
            self.model.cuda()

    def update(self):
        """Update the model weights if there is a new version of the weights file"""
        model_updated = os.path.getmtime(self.model_path)
        if model_updated > self.model_last_updated:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            if self.use_gpu:
                self.model.cuda()
            self.model_last_updated = model_updated

    def processHeatMap(self, hm, cmap = plt.get_cmap('jet')):
        resize_transform = T.Compose([T.ToPILImage()])
        hm = torch.Tensor(hm)
        hm = np.uint8(cmap(np.array(hm)) * 255)
        return hm

    def postprocess(self, pred, threshold=100):
        """
        Runs the depth image through the model.
        Returns the dense prediction of corners, outer edges, inner edges, and a three-channel image with all three.
        """
        # pred = np.load('/media/ExtraDrive1/clothfolding/test_data/pred_62_19_01_2020_12:53:16.npy')

        corners = self.processHeatMap(pred[:, :, 0])
        outer_edges = self.processHeatMap(pred[:, :, 1])
        inner_edges = self.processHeatMap(pred[:, :, 2])

        corners = corners[:,:,0]
        corners[corners<threshold] = 0
        corners[corners>=threshold] = 255

        outer_edges = outer_edges[:,:,0]
        outer_edges[outer_edges<threshold] = 0
        outer_edges[outer_edges>=threshold] = 255

        inner_edges = inner_edges[:,:,0]
        inner_edges[inner_edges<threshold] = 0
        inner_edges[inner_edges>=threshold] = 255

        return corners, outer_edges, inner_edges

    def predict(self, image):
        # 切换成评估模式,关闭dropout和batch normalization等
        self.model.eval()
        # N: 获取 Config， 进行图像裁剪，裁剪的标准是什么？(255,243)的图像也需要再裁剪吗？
        print("image.shape.before: ", image.shape)
        row_start, row_end, col_start, col_end, step = self.crop_dims
        image = image[row_start:row_end:step, col_start:col_end:step]
        print("image.shape.after: ", image.shape)

        # 找到深度图像最大值，将 NAN 也替换成最大值防止出错
        max_d = np.nanmax(image)
        image[np.isnan(image)] = max_d
        # 将 NumPy 数组转换为 PIL 图像对象
        img_depth = Image.fromarray(image, mode='F')
        # 将 PIL 图像对象转换为 Pytorch 张量对象
        img_depth = self.transform(img_depth)

        min_I = img_depth.min()
        max_I = img_depth.max()
        img_depth[img_depth<=min_I] = min_I
        # 所有像素归一化，都在 0-1 之间
        img_depth = (img_depth - min_I) / (max_I - min_I)
        # 与 PyTorch 中的张量格式相对应，将图像张量添加一个维度 (batch size 维度)，以便于将其输入到模型中。
        # 该维度的大小为 1，因为这里只处理了一张图像
        img_depth = img_depth[np.newaxis, :]

        if self.use_gpu:
            inputs = Variable(img_depth.cuda())
        else:
            inputs = Variable(img_depth)

        outputs = self.model(inputs)
        outputs = torch.sigmoid(outputs)
        output = outputs.data.cpu().numpy()
        # 获取输出数组的形状，其中 N 为 batch size，h 和 w 分别为图像的高度和宽度
        N, _, h, w = output.shape
        # 将 output 进行转置，将通道数移动到最后一个维度，即将输出结果转换为 (h, w, C) 的形式，其中 C 表示通道数。
        # 由于上面已经获取了 batch size，所以这里取出第一个图像的预测结果，即 output[0]
        pred = output.transpose(0, 2, 3, 1)
        pred = pred[0]

        corners, outer_edges, inner_edges = self.postprocess(pred)
        return corners, outer_edges, inner_edges, pred

if __name__ == '__main__':
    im = np.load('/media/ExtraDrive1/clothfolding/data_towel_table/camera_0/depth_467_31_12_2019_12:51:20:230966.npy')
    m = ClothEdgeModel()
    out = m.predict(im)
    # print(out)
