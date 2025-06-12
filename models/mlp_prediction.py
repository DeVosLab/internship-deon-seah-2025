from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Split data into target and features, then train/val/test sets on image level
def split_data(input_path, intensity_threshold):
    data = pd.read_parquet(input_path,
                           engine='fastparquet')

    # create labels to identify positive/negative cells
    imgs = data.groupby('filename_img')['intensity_1'].apply(lambda x: (x >= intensity_threshold).mean()).reset_index()
    imgs['label'] = (imgs['intensity_1'])
    # binarise intensity_1 labels
    data['label'] = (data['intensity_1'] >= intensity_threshold).astype(int)

    # first split: train/(valid+test) 70/30
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_idx, temp_idx in sss.split(data, data['label']):
        train = data.iloc[train_idx]
        temp = data.iloc[temp_idx]

    # second split: valid/test 10/20
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=2/3, random_state=42)
    for val_idx, test_idx in sss2.split(temp, temp['label']):
        valid = temp.iloc[val_idx]
        test = temp.iloc[test_idx]

    filename_coords = test[['filename_img', 'z', 'y', 'x', 'intensity_1']].values

    ## split on an image level
    #train = data[data['filename_img'].isin(train)].reset_index(drop=True)
    #valid = data[data['filename_img'].isin(valid)].reset_index(drop=True)
    #test = data[data['filename_img'].isin(test)].reset_index(drop=True)

    return data, train, valid, test, filename_coords

# Process features and target into dataloaders
def process_features(input_path, on_channel, intensity_threshold):
    data, train, valid, test, filename_coords = split_data(input_path, intensity_threshold)

    # extract features & target by channel
    features = data.iloc[:, 11:]
    ft_0 = features.columns[0:1024].tolist()
    ft_1 = features.columns[1024:2048].tolist()

    # splitting features from target by sets
    ft = ft_0 if on_channel == 0 else ft_1
    X_train = train[ft].values
    y_train = train['intensity_1'].values
    X_valid = valid[ft].values
    y_valid = valid['intensity_1'].values
    X_test = test[ft].values
    y_test = test['intensity_1'].values

    # scale targets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # binarise y values
    y_train_bin = (y_train >= intensity_threshold).astype(int)
    y_valid_bin = (y_valid >= intensity_threshold).astype(int)
    y_test_bin = (y_test >= intensity_threshold).astype(int)

    # convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled,
                                  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_bin,
                                  dtype=torch.float32).reshape(-1, 1)
    X_valid_tensor = torch.tensor(X_valid_scaled,
                                  dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid_bin,
                                  dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled,
                                 dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_bin,
                                 dtype=torch.float32).reshape(-1, 1)
    
    # create dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True)
    valid_loader = DataLoader(
        TensorDataset(X_valid_tensor, y_valid_tensor),
        batch_size=32)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=32)
    
    # calculate pos_weight for class balancing
    num_pos = (y_train_tensor == 1).sum().item()
    num_neg = (y_train_tensor == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos],
                            dtype=torch.float32)

    return train_loader, valid_loader, test_loader, pos_weight, filename_coords

# Initialise network parameters
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

def main(**kwargs):

    assert kwargs['input_path'] is not None
    assert kwargs['output_path'] is not None
    assert kwargs['intensity_thresh'] is not None

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_path = kwargs['input_path']
    output_path = kwargs['output_path']
    Path(output_path).mkdir(exist_ok=True, parents=True)
    intensity_threshold = kwargs['intensity_thresh']
    on_channel = kwargs['on_channel']
    sample_confirm = kwargs['sample_confirm']

    train, valid, test, pos_weight, filename_coords = process_features(input_path, on_channel, intensity_threshold)

    print('Epoch\tTrain loss\tValid loss\tValid acc. (%)')

    model = MLP()
    optimiser = torch.optim.Adam(model.parameters(),
                                lr=0.001,
                                weight_decay=1e-5)
    scheduler = StepLR(optimiser,
                    step_size=5,
                    gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    valid_acc_list = []

    epochs = 50
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        train_losses = []
        valid_losses = []

        # train the model
        for images, labels in tqdm(train, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimiser.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels.view(-1, 1))
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0

        # validate the model
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid):
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
        print(f'{epoch+1}\t{np.mean(train_losses):.4f}\t\t{np.mean(valid_losses):.4f}\t\t{accuracy:.2f}')

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

        scheduler.step()

    # assess model's performance on test set        
    model.load_state_dict(torch.load('best_model.pt'))    
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test:
            outputs = model(images)
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

        print('\nConfusion matrix for test set:')
        print(confusion_matrix(all_labels, all_preds, normalize='true'),'\n')

        fig_dir = Path(output_path)
        fig_dir.mkdir(exist_ok=True, parents=True)
        # compute ROC and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr,
                label=f'FOC curve (AUC = {auc:.2f})',
                color='darkorange')
        plt.plot([0, 1], [0, 1],
                'k--',
                label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic, ROC on Channel {on_channel}')
        plt.legend(loc='lower right')
        plt.grid(True)

        fig_path = f'{fig_dir}/{time_stamp}_MLP_ROC_AUC_channel_{on_channel}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved ROC!')

        plt.tight_layout()

    # Merge with original test data
    if sample_confirm is not None:
        print('Saving true labels, predicted probabilities, and predicted class for all samples...')
    else:
        print(f'Saving true labels, predicted probabilities, and predicted class for {sample_confirm}...')
    test_data = pd.DataFrame(filename_coords[:, 0:5], columns=['filename_img', 'z', 'y', 'x', 'intensity_1'])
    test = test_data[['intensity_1']].values
    test_bin = (test >= intensity_threshold).astype(int)
    test_data['true_labels'] = test_bin
    test_data['predicted_probability'] = all_probs
    test_data['predicted_class'] = all_preds

    if sample_confirm is not None:
        sample = sample_confirm
        test_data = test_data[test_data['filename_img'].apply(lambda p: Path(p).stem == sample)]

    # Save predictions to CSV
    csv_path = f'{fig_dir}/{time_stamp}_test_predictions_channel_{on_channel}.csv'
    test_data.to_csv(csv_path, index=False)
    print('Saved predictions!')

    plt.show()

def parse_args():
    parser = ArgumentParser(description='Class Label Prediction using SVC model')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Path to input file with features')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output folder')
    parser.add_argument('--on_channel', type=int, default=0,
                        help='Channel on which prediction is performed')
    parser.add_argument('--intensity_thresh', type=int, default=None,
                        help='Define the intensity threshold for binary classification')
    parser.add_argument('--sample_confirm', type=str, default=None,
                        help='If unspecified, all data will be saved to a CSV file, if specified, only data for selected sample will be saved.')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))