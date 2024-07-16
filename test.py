import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import  confusion_matrix, accuracy_score, classification_report
from network import C3D_model
from dataloaders.dataset import VideoDataset

# Use GPU if available else revert to CPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled=False
print("Device being used:", device)

dataset = 'ucf101'  # Dataset name
num_classes = 3     # Number of classes

save_dir = '/share/luoqifeng-local/attribution_analysis/run/run_2/models/C3D-ucf101_epoch-99.pth.tar'  # Directory where the checkpoint is saved


model = C3D_model.C3D(num_classes=num_classes, pretrained=False)


# Load the checkpoint
checkpoint = torch.load(save_dir, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

# Move the model to the device
model.to(device)
model.eval()

# Load validation data
val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=32),batch_size=32, num_workers=4)

# Lists to store predictions and labels
all_preds = []
all_labels = []

# Loop over the validation data

with torch.no_grad():
    for inputs, labels in tqdm(val_dataloader):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        outputs = model(inputs)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        

accuracy = accuracy_score(all_labels, all_preds)
print(f'Overall Accuracy: {accuracy:.6f}')
