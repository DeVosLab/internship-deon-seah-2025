import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import warnings

warnings.filterwarnings('ignore')

# Read in data from .parquet/.h5 file
print('Reading data from parquet file...')
#data = pd.read_parquet('D:/3D_data/output/embeddings/1-60-60_mae_cell_embeddings.parquet',
#                     engine='fastparquet')
data = pd.read_parquet('~/projects/deon/3D_data/embeddings/1-60-60_mae_cell_embeddings.parquet',
                     engine='fastparquet')

# Split data into target and features, then train/val/test sets on image level
print('Splitting data into training, validation, and test sets by image groups...')
imgs = data.groupby('filename_img')['intensity_1'].apply(lambda x: (x >= 2500).mean()).reset_index()
imgs['label'] = (imgs['intensity_1'])

# Binarise intensity_1 labels
data['label'] = (data['intensity_1'] >= 2500).astype(int)

# First split: training/valid+test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in sss.split(data, data['label']):
    train = data.iloc[train_idx]
    temp = data.iloc[temp_idx]

# Second split: valid/test from 30%
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=42)
for val_idx, test_idx in sss2.split(temp, temp['label']):
    valid = temp.iloc[val_idx]
    test = temp.iloc[test_idx]

## Split on an image level
#train = data[data['filename_img'].isin(train_imgs)].reset_index(drop=True)
#valid = data[data['filename_img'].isin(valid_imgs)].reset_index(drop=True)
#test = data[data['filename_img'].isin(test_imgs)].reset_index(drop=True)

# Extract features and target from each split
print('Splitting data into features and target from each split...')
ft = [col for col in data.columns if 'embedding' in col]
X_train = train[ft].values
y_train = train['intensity_1'].values
X_valid = valid[ft].values
y_valid = valid['intensity_1'].values
X_test = test[ft].values
y_test = test['intensity_1'].values

# Scale training set
print('Scaling X_train and X_test...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Binarise y values
print('Binning y values...')
y_train_bin = (y_train >= 2500).astype(int)
y_valid_bin = (y_valid >= 2500).astype(int)
y_test_bin = (y_test >= 2500).astype(int)

# Convert to PyTorch tensors
print('Converting values to PyTorch tensors...')
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_bin, dtype=torch.float32).reshape(-1, 1)
X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid_bin, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_bin, dtype=torch.float32).reshape(-1, 1)

# Create DataLoaders
print('Creating DataLoaders...')
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=64,
    shuffle=True)
valid_loader = DataLoader(
    TensorDataset(X_valid_tensor, y_valid_tensor),
    batch_size=64)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=64)

# Initialise network parameters
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Linear(100, 1))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
model = MLP()

optimiser = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 5

for epoch in range(epochs):
    model.train()

    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):
        optimiser.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels.view(-1, 1))
        loss.backward()
        optimiser.step()

        train_losses.append(loss.item())

        if (i * 128) % (128 * 100) == 0:
            print(f'{i * 128} / 50000')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            outputs = model(images)
            loss = loss_fn(outputs, labels.view(-1, 1))
            valid_losses.append(loss.item())

            predicted = (outputs > 0.5).int().view(-1)
            labels = labels.view(-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))

    accuracy = 100*correct/total
    valid_acc_list.append(accuracy)
    print(f'Epoch: {epoch+1}, train loss: {np.mean(train_losses):.4f}, valid loss: {np.mean(valid_losses):.4f}, valid acc.: {accuracy:.2f}%')

    model.eval()
    test_preds = torch.LongTensor()

    for images, _ in test_loader:
        outputs = model(images)
        pred = (outputs > 0.5).int()
        test_preds = torch.cat((test_preds, pred), dim=0)

    out_df = pd.DataFrame()
    out_df['ID'] = np.arange(1, len(X_test)+1)
    out_df['label'] = test_preds.numpy()

    out_df.head()