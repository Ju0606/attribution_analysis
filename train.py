import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model

from sklearn.metrics import confusion_matrix
import numpy as np




#混淆矩阵
def save_confusion_matrix_to_txt(cm, class_names, epoch, save_dir,phase):
    filename = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}_{phase}.txt')
    with open(filename, 'w') as f:
        f.write(f'Confusion matrix for epoch {epoch}:\n\n')
        f.write(' '.join(class_names) + '\n\n')
        for i, row in enumerate(cm):
            f.write(f'{class_names[i]} ' + ' '.join(map(str, row)) + '\n')
    print(f'Confusion matrix for epoch {epoch} saved to {filename}')


# Use GPU if available else revert to CPU
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = False #True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-5 # 1e-3Learning rate
#torch.backends.cudnn.enabled=False #在环境C3D2中禁用cudnn

dataset = 'ucf101' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 3
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    
    train_dataset = VideoDataset(dataset='ucf101', split='train', clip_len=32, preprocess=False)
    #batch_size=20, shuffle=False
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=32,preprocess=False), batch_size=32, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=32), batch_size=32, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=32), batch_size=32, num_workers=4)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    print(tqdm(trainval_loaders['train']))
    
    #class_weights = compute_class_weight('balanced', classes=np.unique(train_dataloader.dataset.labels), y=train_dataloader.dataset.labels)
    #class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    #class_weights = train_dataset.compute_class_weights().to('cuda')
    class_weights = train_dataset.compute_class_weights().to(device)
    print('class_weights:',class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    #criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    #optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    '''
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16,preprocess=False), batch_size=20, shuffle=False, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=20, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=20, num_workers=4)
    
        
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    '''
    
    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            #混淆矩阵
            all_labels = []
            all_preds = []
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                #print(len(inputs))
                if phase == 'train':
                    outputs = model(inputs)
                    
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #增加混淆矩阵部分
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
            print('trainval_sizes:',trainval_sizes)
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                
                
            #混淆矩阵
            if num_epochs-epoch<=10:#epoch%10==9:
                cm = confusion_matrix(all_labels, all_preds)
                save_confusion_matrix_to_txt(cm, class_names=[str(i) for i in range(num_classes)], epoch=epoch, save_dir=save_dir,phase=phase)
            
            
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()
    


if __name__ == "__main__":
    train_model()