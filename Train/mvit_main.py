import sys
sys.path.append("../Model/")
sys.path.append("../Utils/")
sys.path.append("../Train/")
import random
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torch import stack, cuda, nn, optim
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import time
import torch
import numpy as np
import warnings
import os
import datetime
from Dataloader_Flow import Flow_loader
from Model_VGGFace import resnet18_pt_mcn, Flow_Part_npic
from MobileViT.load_model import load_mobilevit_weights
import torch.nn.functional as F
from collections import Counter
import timm
from transformers import get_linear_schedule_with_warmup
# 定义学习率调度函数
def lr_lambda(current_step):
    print(current_step)
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(EPOCH - current_step) / float(max(1, EPOCH - warmup_steps)))
# 环境变量和随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
VERSION = ''#'_balanced' #
from sklearn.utils.class_weight import compute_class_weight



class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha.cuda()
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()


warnings.filterwarnings("ignore")

# 设置日志文件
times = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')
log_file = open(f'{times}.txt', 'w')

random.seed(1)

# 超参数
super_para = {"LEARNING_RATE": 0.005, "FOLD": 156, "BATCH_SIZE": 64, "EPOCH": 300, "WEIGHT_DECAY": 1e-4,
              'Clip_Norm': 1, 'Micro': "SAMM, CASME, SMIC,CASME3", 'Macro': "CK",
              'Sample_File': '../Sample_File/VGG_2.txt', 'Num_Workers': 4, 'warmup_steps': 1000}

LEARNING_RATE = super_para["LEARNING_RATE"]
BATCH_SIZE = super_para["BATCH_SIZE"]
EPOCH = super_para["EPOCH"]
Clip_Norm = super_para['Clip_Norm']
Sample_File = super_para['Sample_File']
FOLD = super_para['FOLD']
Num_Workers = super_para['Num_Workers']
WEIGHT_DECAY = super_para["WEIGHT_DECAY"]
warmup_steps = super_para['warmup_steps']

# 打印并记录超参数
def log(message, printf = True):
    if printf:
        print(message)
    log_file.write(message + '\n')

log(str(super_para))
log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

parser = argparse.ArgumentParser()
# parser.add_argument('--Micro_label', default='../combined_label_balanced.txt')
# parser.add_argument('--Micro_subject', default='../combined_subject_balanced.txt')
# parser.add_argument('--Micro_merge', default='../combined_afflow_balanced.txt')
# parser.add_argument('-E', '--Epoch', default=EPOCH, type=int)
parser.add_argument('--Micro_label', default=f'../combined_label{VERSION}.txt')
parser.add_argument('--Micro_subject', default=f'../combined_subject{VERSION}.txt')
parser.add_argument('--Micro_merge', default=f'../combined_afflow{VERSION}.txt')
parser.add_argument('-E', '--Epoch', default=EPOCH, type=int)
# 保存模型状态字典的函数
def save_model_state(model, file_path):
    state_dict = {name: sub_model.state_dict() for name, sub_model in model.items()}
    torch.save(state_dict, file_path)
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, iter_num, alpha=10, low=0.0, high=1.0, max_iter=10000.0):
        ctx.iter_num = iter_num
        ctx.alpha = alpha
        ctx.low = low
        ctx.high = high
        ctx.max_iter = max_iter
        output = input.clone()
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        coeff = np.float(2.0 * (ctx.high - ctx.low) / (1.0 + np.exp(-ctx.alpha * ctx.iter_num / ctx.max_iter)) - (ctx.high - ctx.low) + ctx.low)
        return -coeff * gradOutput, None, None, None, None, None




# ReverseLayerF = ReverseLayerF()


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

def calculate_alpha(num_classes, class_counts):
    total_samples = sum(class_counts)
    alpha = [total_samples / count for count in class_counts]
    # Normalize alpha so that sum(alpha) = num_classes
    alpha = [a / sum(alpha) * num_classes for a in alpha]
    print(alpha)
    return torch.tensor(alpha, dtype=torch.float32)



def train(train_dataloader, model, criterion, optimizer, scheduler, epoch, print_freq=10):

    model['mvit'].train()
    model['discriminator'].train()

    correct = 0

    with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCH}', unit='batch') as pbar:
        for i, sample in enumerate(train_dataloader):
            input, label, domain_label = sample['image'], sample['label'], sample['domain_label']
            input, label, domain_label = input.cuda(), label.cuda(), domain_label.cuda()

            output,features = model['mvit'](input)

            # reversed_features = ReverseLayerF(features)
            # Example usage:
            iter_num = len(train_dataloader) * epoch
            reversed_features = ReverseLayerF.apply(features, iter_num)

            output_domain = model['discriminator'](reversed_features).squeeze()

            loss_domain = criterion['domain'](output_domain, domain_label.float())

            loss_label = criterion['label'](output, label)

            loss = loss_label +  loss_domain

            _, preds = torch.max(output, dim=1)

            correct += (label.int() == preds.int()).sum().item()
            accuracy = correct / ((i + 1) * BATCH_SIZE)
            

            
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model['mvit'].parameters(), Clip_Norm, norm_type=2)

            optimizer.step()
            scheduler.step()

            if i % print_freq == 0:
                pbar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy})
                log(f'Train Epoch:[{epoch}][{i}/{len(train_dataloader)}] '
                    f'Acc: {accuracy:.3f} Loss: {loss.item():.4f}', printf = False)
            pbar.update(1)


def validate(validate_dataloader, model, criterion, epoch):

    model['mvit'].eval()

    losses = 0
    correct = 0
    preds_return = torch.LongTensor([])
    target_return = torch.LongTensor([])
    sample_file = {}
  
    with torch.no_grad():

        for sample in validate_dataloader:
            input, target, file_name = sample['image'], sample['label'], sample['file_name']
            input, target = input.cuda(), target.cuda()

            output,_ = model['mvit'](input)



            loss = criterion['label'](output, target)

            _, preds = torch.max(output, dim=1)

            preds_return = torch.cat((preds_return, preds.cpu()), 0)
            target_return = torch.cat((target_return, target.cpu()), 0)

            losses += loss.item()
            correct += (preds == target).sum().item()

            for f in range(len(file_name[0])):
                sample_file[file_name[0][f]] = [int(preds[f].item()), int(target[f].item())]

    accuracy = correct / len(validate_dataloader.dataset)
    return preds_return, target_return, losses, accuracy, sample_file


def build_model_2pic(num_classes=7):

    model = Flow_Part_npic(num_pic=2, num_classes=num_classes)

    model_resnet = resnet18_pt_mcn(weights_path='../Model/resnet18_pt_mcn.pth')
    pretrained_dict = model_resnet.state_dict()
    model_dict = model['resnet'].state_dict()
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    model['resnet'].load_state_dict(model_dict)

    return model
def load_model_state(model, file_path):
    state_dict = torch.load(file_path)
    for name, sub_model in model.items():
        sub_model.load_state_dict(state_dict[name])
def confusion_matrix_heatmap(name, matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 创建一个热图
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels',fontsize=15, color='k') #x轴label的文本和字体大小
    plt.ylabel('True labels',fontsize=15, color='k') #y轴label的文本和字体大小
    plt.xticks(fontsize=10) #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=10) #y轴刻度的字体大小（文本包含在pd_data中了）
    plt.title(name+' Confusion Matrix',fontsize=20) #图片标题文本和字体大小

    # 保存图像
    plt.savefig(name+f"_confusion_matrix{VERSION}.png")
def filter_micro_elements(original_list):
    # 初始化一个新列表用于存放包含'Micro'的元素
    micro_elements = []
    # 使用列表推导式和条件检查来过滤和移动元素
    micro_elements = [element.split('_')[0] for element in original_list if 'Micro' in element]
    # 从原始列表中移除包含'Micro'的元素
    original_list = [element for element in original_list if 'Micro' not in element]
    # 返回修改后的原始列表和新列表
    return original_list, micro_elements
def filter_test_elements(test_list,micro_list):
    test_list = [element+'_Micro' for element in micro_list if element not in test_list]
    return test_list
def main():
    args = parser.parse_args()
    Micro_label = args.Micro_label
    Micro_subject = args.Micro_subject
    Micro_merge = args.Micro_merge

    start_time = time.time()
        
    class_counts = [938, 130, 1545, 1299, 970, 1119, 816]  # 示例数据
    num_classes = len(class_counts)

    # alpha = calculate_alpha(num_classes, class_counts)
    alpha = torch.tensor([5,6,0.2,0.5,0.5,0.8,0.5], dtype=torch.float32)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomCrop((256, 256), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    with open(Micro_subject, 'r') as s:
        subject_origin = s.readlines()
    with open(Micro_label, 'r') as l:
        label_origin = l.readlines()
    
    label_origin = [label.strip('\n') for label in label_origin]
    subject_origin = [subject.strip('\n') for subject in subject_origin]
    index_length = len(subject_origin)
    subject = set(subject_origin)
    subject = list(subject)
    subject.remove('Macro')
    subject.remove('Micro')
    subject, micro_subject = filter_micro_elements(subject)
    data_array = np.arange(index_length)

    Fold_accuracy = 0
    TP_Fold, TN_Fold, FN_Fold, FP_Fold = {}, {}, {}, {}
    FOLD = len(subject)
    log('FOLD total: ' + str(len(subject)))
    train_index, test_index = train_test_split(np.arange(len(subject)), test_size=0.25, random_state=42)

    with open('sample_file.txt', 'w') as file_handle:
        file_handle.write('Train/Test split with 8:2 ratio\n')
        
        train_subjects = [subject[i] for i in train_index]
        train_subjects.append('Macro')
        train_subjects.append('Micro')
        test_subjects = [subject[i] for i in test_index]
        train_extend = filter_test_elements(test_subjects,micro_subject)
        train_subjects.extend(train_extend)
        
        log('Train subjects: ' + str(train_subjects) + ' ' + str(len(train_subjects)))
        log('Test subjects: ' + str(test_subjects) + ' ' + str(len(test_subjects)))

        with open(Sample_File, 'a') as S:
            S.writelines(str(test_subjects[0]) + '\n')

        # 获取训练和测试索引
        train_index = [data_array[index] for index in range(index_length) if subject_origin[index] in train_subjects]
        test_index = [data_array[index] for index in range(index_length) if subject_origin[index] in test_subjects]
        log('Train length: ' + str(len(train_index)))
        log('Test length: ' + str(len(test_index)))
        train_label = [label_origin[index] for index in train_index]
        test_label = [label_origin[index] for index in test_index]
        label_train_counts = Counter(train_label)
        label_test_counts = Counter(test_label)

        # 计算类权重
        class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
        class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

        

        log('Train label counts: ' + str(label_train_counts))
        log('Test label counts: ' + str(label_test_counts))
        # 创建训练和测试数据集
        train_dataset = Flow_loader(Micro_merge, Micro_label, Micro_subject, train_index, transform=transform)
        test_dataset = Flow_loader(Micro_merge, Micro_label, Micro_subject, test_index, transform=transform)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=Num_Workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=Num_Workers)
        model = {}
        model['mvit'] = load_mobilevit_weights('S')
        

        model['discriminator'] = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        model['mvit'] = model['mvit'].cuda()       
        model['discriminator'] = model['discriminator'].cuda()
        model['discriminator'] = model['discriminator'].apply(weight_init)

        criterion = {
            # 假设有 7 个类别，各类别样本数量如下
            # 'label': FocalLoss(alpha=alpha), #alpha=[0.1, 0.2, 0.3, 0.15, 0.25]
            'label': torch.nn.CrossEntropyLoss(),#weight=class_weights),
            'domain': torch.nn.BCEWithLogitsLoss()
        }

        optimizer = optim.Adam([
            {'params': model['mvit'].parameters(), 'lr': LEARNING_RATE},
            {'params': model['discriminator'].parameters(), 'lr': LEARNING_RATE},
        ], weight_decay=WEIGHT_DECAY)

        Epoch_accuracy = 0
        Epoch_F1_score = 0
        Epoch_Recall = 0
        log('Start train')
        total_steps = len(train_loader) * EPOCH  # 添加
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)  # 添加
        # scheduler = CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)  # 添加

        for epoch in range(args.Epoch):
            # if epoch % 10 == 0 and epoch != 0:
            #     if optimizer.param_groups[1]['lr'] > 0.00001:
            #         optimizer.param_groups[1]['lr'] *= 0.5
            
            train(train_loader, model, criterion, optimizer,scheduler, epoch)
            preds, target, loss, accuracy, temp_file = validate(test_loader, model, criterion, epoch)
            
            accuracy = accuracy_score(target, preds)
            matrix = confusion_matrix(target, preds, labels=[0,1,2,3,4,5,6])#['anger','contempt','disgust', 'fear','happy', 'sad', 'surprise'])
            FP = matrix.sum(axis=0) - np.diag(matrix)
            FN = matrix.sum(axis=1) - np.diag(matrix)
            TP = np.diag(matrix)
            TN = matrix.sum() - (FP + FN + TP)

            f1_s = np.ones([7])
            deno = (2 * TP + FP + FN)
            for f in range(7):
                if deno[f] != 0:
                    f1_s[f] = (2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f])
                else:
                    f1_s[f] = 1

            f1 = np.mean(f1_s)
            losses = loss / len(test_index)

            if f1 >= Epoch_F1_score:
                Epoch_F1_score = f1
                Epoch_accuracy = accuracy
                FP_Epoch = FP
                FN_Epoch = FN
                TP_Epoch = TP
                TN_Epoch = TN
                sample_file = temp_file
                save_model_state(model, f'best_model_f1{VERSION}.pth')
                log(f'New best F1 score: {Epoch_F1_score:.4f}, saving model.')
                confusion_matrix_heatmap('Best_F1', matrix)

            if accuracy > Epoch_Recall:
                Epoch_Recall = accuracy
                save_model_state(model, f'best_model_acc{VERSION}.pth')
                log(f'New best accuracy: {Epoch_Recall:.4f}, saving model.')
                confusion_matrix_heatmap('Best_ACC', matrix)

                

            log(f'Validation: Acc: {accuracy:.3f} F1_Score: {f1:.3f} Loss: {losses:.3f} Best F1: {Epoch_F1_score:.3f}')

        with open(Sample_File, 'a') as S:
            for key in sample_file:
                S.writelines(f"{key}\t{sample_file[key][0]}\t{sample_file[key][1]}\n")

        TP_Fold['total'] = TP_Epoch
        TN_Fold['total'] = TN_Epoch
        FN_Fold['total'] = FN_Epoch
        FP_Fold['total'] = FP_Epoch

        Fold_accuracy += Epoch_accuracy

        f1 = np.ones([7])
        deno = (2 * TP_Fold['total'] + FP_Fold['total'] + FN_Fold['total'])
        for f in range(7):
            if deno[f] != 0:
                f1[f] = (2 * TP_Fold['total'][f]) / deno[f]
            else:
                f1[f] = 1

        current_f1 = np.mean(f1)

        log(f'Runned Subject: F1 Score: {current_f1:.3f}')

    F1_Score = {'total': (2 * TP_Fold['total']) / (2 * TP_Fold['total'] + FP_Fold['total'] + FN_Fold['total'])}
    Recall_Score = {'total': TP_Fold['total'] / (TP_Fold['total'] + FN_Fold['total'])}

    Total_accuracy = Fold_accuracy
    Total_F1_Score = {'total': np.mean(F1_Score['total'])}
    Total_Recall = {'total': np.mean(Recall_Score['total'])}

    log(f'Fold accuracy: {Total_accuracy}')
    log(f'Fold F1 score: {Total_F1_Score["total"]}')
    log(f'Fold Recall score: {Total_Recall["total"]}')

    end_time = time.time()
    log(f'Total cost time: {end_time - start_time}')

    log_file.close()

if __name__ == '__main__':
    main()

