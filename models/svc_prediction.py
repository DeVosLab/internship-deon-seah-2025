from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pandas as pd
from umap import UMAP
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Split data into target and features, then train/val/test sets on image level
def split_data(input_path):
    data = pd.read_parquet(input_path,
                           engine='fastparquet')

    # create labels to identify positive/negative cells
    imgs = data.groupby('filename_img')['intensity_1'].apply(lambda x: (x >= 2500).mean()).reset_index()
    imgs['label'] = (imgs['intensity_1'])
    # binarise intensity_1 labels
    data['label'] = (data['intensity_1'] >= 2500).astype(int)

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

    ## split on an image level
    #train = data[data['filename_img'].isin(train)].reset_index(drop=True)
    #valid = data[data['filename_img'].isin(valid)].reset_index(drop=True)
    #test = data[data['filename_img'].isin(test)].reset_index(drop=True)

    return data, train, valid, test

# Process features and target
def process_features(input_path, on_channel):
    data, train, valid, test = split_data(input_path)

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
    y_train_bin = (y_train >= 2500).astype(int)
    y_valid_bin = (y_valid >= 2500).astype(int)
    y_test_bin = (y_test >= 2500).astype(int)

    # initialise UMAP model
    umap_model = UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.3,
        metric='euclidean',
        random_state=42)

    # reduce dimensions for all features    
    X_train_reduced = umap_model.fit_transform(X_train)
    X_valid_reduced = umap_model.transform(X_valid)
    X_test_reduced = umap_model.transform(X_test)

    return X_train_reduced, X_valid_reduced, X_test_reduced, y_train_bin, y_valid_bin, y_test_bin

# Train SVM model
def main(**kwargs):
    assert kwargs['input_path'] is not None
    assert kwargs['output_path'] is not None

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    input_path = kwargs['input_path']
    output_path = kwargs['output_path']
    Path(output_path).mkdir(exist_ok=True, parents=True)
    on_channel = kwargs['on_channel']

    X_train, X_valid, X_test, y_train, y_valid, y_test = process_features(input_path, on_channel)

    # initialise SVC model
    svc_model = SVC(
        kernel='rbf',
        gamma='auto',
        C=10,
        class_weight='balanced',
        probability=True)
    
    # fit model to training set
    svc_model.fit(X_train, y_train)

    # create predictions on validation set
    val_predict = svc_model.predict(X_valid)
    prediction = svc_model.predict(X_test)

    print('Classification report for validation set:')
    print(classification_report(y_valid, val_predict))

    print('Confusion matrix for test set:')
    print(confusion_matrix(y_test, prediction, normalize='true'))

    # get predicted probabilities for test set
    y_scores = svc_model.predict_proba(X_test)[:, 1]

    fig_dir = Path(output_path)
    fig_dir.mkdir(exist_ok=True, parents=True)
    # compute ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

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
    plt.title(f'Receiver Operating Characteristic, ROC on Channel {on_channel}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    fig_path = f'{fig_dir}\\{time_stamp}_SVC_ROC_AUC_on_channel_{on_channel}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved ROC!')

    plt.tight_layout()
    plt.show()

def parse_args():
    parser = ArgumentParser(description='Class Label Prediction using SVC model')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Path to input file with features')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output folder')
    parser.add_argument('--on_channel', type=int, default=0,
                        help='Channel on which prediction is performed')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))