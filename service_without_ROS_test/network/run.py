import torch 
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from unet import unet
from utils import normalize

class Run:
    def __init__(self, model_path, n_features=3):
        self.model_path = model_path
        self.n_features = n_features
        self.load_model()

    def load_model(self):
        print("in init & load model, self.n_features =", self.n_features)
        # construct U-Net model, n_classes代表分类特征数，in_channels
        self.model = unet(n_classes=self.n_features, in_channels=1)
        
        self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            torch.cuda.device(0)
            self.model = self.model.cuda()

    # model evaluate
    def evaluate(self, depth):
        # 在训练模式下，模型会保存并计算梯度，从而对模型参数进行更新。
        # 在评估模式下，模型不会计算梯度，而是直接对输入进行前向传播并输出结果。
        self.model.eval()
        img_depth = Image.fromarray(depth, mode='F')
        
        transform = T.Compose([T.ToTensor()])
        img_depth = transform(img_depth)
        img_depth = normalize(img_depth)

        inputs = img_depth.unsqueeze_(0)
        print("model.input.shape:", inputs.shape)
        # model.input.shape: torch.Size([1, 1, 255, 243])
            # batch_size, channel, height, width
        inputs = Variable(inputs.cuda()) if self.use_gpu else Variable(inputs)
        outputs = self.model(inputs)
        print("model.outputs.shape:", outputs.shape)
        # model.outputs.shape: torch.Size([1, 3, 255, 243])
            # batch_size, channel, height, width
        outputs = torch.sigmoid(outputs)
        output = outputs.data.cpu().numpy()
        pred = output.transpose(0, 2, 3, 1)
        print("model.pred.shape", pred.shape)
        # model.pred.shape: (1, 255, 243, 3)
            # batch_size, height, witdth, channel
        return pred