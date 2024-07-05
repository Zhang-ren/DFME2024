import torch
import sys
sys.path.append("../Model/")
sys.path.append("../Utils/")
sys.path.append("../Train/")
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Model_VGGFace import resnet18_pt_mcn, Flow_Part_npic
from MobileViT.load_model import load_mobilevit_weights
from torch import nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class TestDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'r') as f:
            self.image_paths = f.readlines()
        self.image_paths = [x.strip() for x in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# 假设你已经定义了构建模型的函数 build_model_2pic() 和加载模型状态的函数 load_model_state()
def build_model_2pic(num_classes=7):

    model = Flow_Part_npic(num_pic=2, num_classes=num_classes)

    return model
def load_model_state(model, file_path):
    state_dict = torch.load(file_path)
    for name, sub_model in model.items():
        sub_model.load_state_dict(state_dict[name])
        sub_model.cuda()
        sub_model.eval()
# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 创建测试集数据加载器
test_dataset = TestDataset('../Mix_DFME_test_10B_afflow_22.txt', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
model_path = '/home/data2/MEGC/MEGC-DFME/Train/best_model_f1_0371.pth'
# model = build_model_2pic()  # 假设 build_model_2pic() 是你定义的模型构建函数
model = {}
model['mvit'] = load_mobilevit_weights()
model['discriminator'] = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
load_model_state(model, model_path)
# model.eval()

# 预测并保存结果
output_file = '/home/data2/MEGC/MEGC-DFME/predictions.txt'
output_file1 = '/home/data2/MEGC/MEGC-DFME/prediction.txt'
with torch.no_grad():
    with open(output_file, 'w') as f:
        with open(output_file1, 'w') as f1:
            for images, img_paths in test_loader:
                images = images.cuda()
                # _, output_resnet_top, output_resnet_bottom = model['resnet'](images)
                outputs,_ = model['mvit'](images)
                # output_fc_top = model['fc_top'](output_resnet_top)
                # output_fc_bottom = model['fc_bottom'](output_resnet_bottom)

                # output_model = torch.cat((output_fc_top, output_fc_bottom), 1)

                # outputs = model['classifier'](output_model)
                _, predicted = torch.max(outputs, 1)
                for img_path, pred in zip(img_paths, predicted):
                    f.write(f'{img_path}\t{pred.item()}\n')
                    f1.write(f'{pred.item()}\n')


