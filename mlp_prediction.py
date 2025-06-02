import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
model = MLP()

optimiser = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5)
loss_fn = nn.BCEWithLogitsLoss()

mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []

epochs = 20
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    model.train()

    train_losses = []
    valid_losses = []

    # train the model
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

    # validate the model
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            outputs = model(images)
            loss = loss_fn(outputs, labels.view(-1, 1))
            valid_losses.append(loss.item())

            predicted = (outputs > 0.5).int().view(-1)
            labels = labels.view(-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = np.mean(valid_losses)
    accuracy = 100*correct/total
    valid_acc_list.append(accuracy)
    print(f'Epoch: {epoch+1}, train loss: {np.mean(train_losses):.4f}, valid loss: {np.mean(valid_losses):.4f}, valid acc.: {accuracy:.2f}%')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'No improvement for {patience_counter} epoch(s).')
        if patience_counter >= patience:
            print('Early stopping triggered.')
            torch.save(model.state_dict(), 'best_model.pt')
            break
        
model.load_state_dict(torch.load('best_model.pt'))    
model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = torch.sigmoid(outputs).view(-1)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print('Confusion matrix for test set:')
    print(confusion_matrix(all_labels, all_preds, normalize='true'))

    accuracy = 100 * correct/total
    print(f'Test accuracy: {accuracy:.2f}%')

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

    fig_path = f'~/projects/deon/3D_data/mlp_output/MLP_ROC_AUC.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')