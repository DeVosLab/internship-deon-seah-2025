import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

"""
o Try a simpler model instead of a neural network--sklearn one-class SVM or SVC
o In one-class SVM: positive cells are anomalies
o In SVC: positive and negative cells are two classes
o Be careful to not overfit on the negative class (majority); model could consider all points negative
o Train on the whole dataset
o Subdivide data on image level first, training 70%, validation 20%, testing 10%
"""

# Read in data from .parquet/.h5 file
print('Reading data from parquet file...')
data = pd.read_parquet('D:/3D_data/output/embeddings/1-60-60_mae_cell_embeddings.parquet',
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
print('Scaling X_train_scaled and X_test_scaled...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Fit PCA model to training set
print('Fitting PCA model to training set...')
pca_model = PCA(
    n_components=250,
    random_state=42)
X_train_reduced = pca_model.fit_transform(X_train_scaled)
X_valid_reduced = pca_model.transform(X_valid_scaled)
X_test_reduced = pca_model.transform(X_test_scaled)

# Binarise y values
y_train_bin = (y_train >= 2500).astype(int)
y_valid_bin = (y_valid >= 2500).astype(int)
y_test_bin = (y_test >= 2500).astype(int)

# Fit SVM model to reduced training set
print('Fitting SVC model to reduced training set...')
svc_model = SVC(
    kernel='rbf',
    gamma='auto',
    C=10,
    class_weight='balanced',
    probability=True)
svc_model.fit(X_train_reduced, y_train_bin)

# Create predictions
#print('Creating predictions on reduced validation set...')
#val_predict = svc_model.predict(X_valid_reduced)
#
#print('Creating predictions on reduced test set...')
#prediction = svc_model.predict(X_test_reduced)
#
#print('Classification report for validation set:')
#print(classification_report(y_valid_bin, val_predict))
#
#print('Classification report for test set:')
#print(classification_report(y_test_bin, prediction))

# Get predicted probabilities for test set
y_scores = svc_model.predict_proba(X_test_reduced)[:, 1]

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test_bin, y_scores)
auc = roc_auc_score(y_test_bin, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr,
    label=f'FOC curve (AUC = {auc:.2f})',
    color='darkorange')
plt.plot(
    [0, 1], [0, 1],
    'k--',
    label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic, ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()