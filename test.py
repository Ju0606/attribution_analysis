import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import  accuracy_score
from network import C3D_model
from dataloaders.dataset import VideoDataset
import os

# 设置显卡
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# 禁用cudnn
torch.backends.cudnn.enabled=False

# 设置数据集
num_classes = 3     

# 设置模型地址和数据集地址
root_dir='/share/luoqifeng-local/AIDE_Dataset'# model和dataset的目录
model_folder='/share/luoqifeng-local/AIDE_Dataset/model'# model文件夹
fold_folder='/share/luoqifeng-local/AIDE_Dataset/fold5'# dataset文件夹
models=os.listdir(model_folder)
folds=os.listdir(fold_folder)
'''
models=['model0.pth.tar', 'model1.pth.tar', 'model2.pth.tar', 'model3.pth.tar', 'model4.pth.tar'] 
folds=['f0', 'f1', 'f2', 'f3', 'f4']
'''

#print('model_dirs:',models)
#print('dataset_dirs:',folds)

# 保存全部的预测结果和原标签
All_preds = []
All_labels = []
# 保存每一折的acc结果
fold_accuracies = []

for fold_idx in range(5):
    print(f"Evaluating fold {fold_idx} ,Using model:{models[fold_idx]},Using dataset:{folds[fold_idx]}")
    model_path=os.path.join(root_dir,'model',models[fold_idx])
    # 加载训练好的模型
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()
    
    
    fold_dir=os.path.join(root_dir,'fold5',folds[fold_idx])
    # 加载测试集
    val_dataloader = DataLoader(VideoDataset(split='val', clip_len=32,output_dir=fold_dir),batch_size=32, num_workers=4)

    # 保存预测结果和原标签
    all_preds = []
    all_labels = []

    # 预测
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            outputs = model(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            All_labels.extend(labels.cpu().numpy())
            All_preds.extend(preds.cpu().numpy())
            
    # 计算每一折的acc并保存
    accuracy = accuracy_score(all_labels, all_preds)
    fold_accuracies.append(accuracy)
    print(f'Fold {fold_idx} Accuracy: {accuracy:.6f}')
    
# 计算所有acc的平均值
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f'Average Accuracy: {average_accuracy:.6f}')

# 计算所有预测结果的acc
Overall_Accuracy = accuracy_score(All_labels, All_preds)
print(f'Overall Accuracy: {Overall_Accuracy:.6f}')
