# Class Label Prediction

## `svc_prediction`: Class label prediction using Support Vector Classification model.
This script performs binary classification using a Support Vector Classifier (SVC) on cell embeddings and metadata stored in the input Parquet file. Based on the intensity threshold specified by the user, it binarises the intensity values of Channel 1 into positive and negative labels, which are used as targets for the classifier. The classifier is trained on the features from the channel that the user specifies, and returns a classification report in the terminal, while saving a ROC plot to the designated output folder.

Support Vector Classification is a supervised machine learning method that aims to find the optimal hyperplane that best separates data points into classes. It works by maximising the margin between the support vectors, which are data points closest to the decision boundary, making the model more robust to outliers. SVC can also handle non-linear data by using kernel tricks to map inputs into higher-dimensional spaces. This makes it especially useful for complex biological data, where class boundaries may not be easily distinguished.

```bash
python -m svc_prediction \
--input_path /path/to/file_with_features.parquet \
--on_channel 0 \
--intensity_thresh 300 \
--output_path /path/to/output_folder
```
Input:
- One Parquet file that contains the cell embeddings and metadata.

Parameters:
- `--input_path`: Path to Parquet file containing cell embeddings and metadata.
- `--output_path`: Path to folder where the ROC plot will be directed to.
- `--on_channel`: Specify channel to predict class labels on.
- `--intensity_thresh`: Specify the intensity threshold to binarise intensity values (from Channel 1) on.

Output:
- Classification report will be returned at the end, on the terminal.
- One PNG file containing the ROC plot.


## `mlp_prediction`: Class label prediction using Multilayer Perceptron model
This script performs class label prediction using a Multilayer Perceptron (MLP) model on cell embeddings and metadata stored in the input Parquet file. It trains the model using channel-specific features, and an intensity threshold, specified by the user, to binarise the target labels. While training and validating the model, the training loss, validation loss, and F1 score for each epoch will be printed onto the terminal, for up to 50 epochs unless early stopping was triggered after 8 consecutive epochs where there was no improvement in the (average) validation loss. A confusion matrix is returned to the terminal at the end, and a ROC plot is saved to the output folder. Optionally, the test-set predictions for a user-specified sample are saved in a CSV file for downstream visual inspection, this includes the ground truth labels, predicted class labels, and predicted probabilities, along with their coordinates. If unspecified, these labels are saved for all samples into a CSV file. These labels can be visualised using the next script `napari_labels_predicted`, allowing for a flexible and interpretable approach to evaluating the classifier's performance across spatially resolved biological data.

A Multilayer Perceptron is a type of feedforward neural network composed of multiple layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node (or neuron) applies a weighted sum of its inputs followed by a non-linear activation function, allowing the network to model complex relationships in the data. During training, the MLP adjusts its weights using backpropagation and gradient descent to minimise the prediction error. MLPs are especially effective for classification tasks, where patterns in high-dimensional data need to be captured and generalised.

```bash
python -m mlp_prediction \
--input_path /path/to/file_with_features.parquet \
--on_channel 0 \
--intensity_thresh 300 \
--sample_confirm 3d_sample_001 \
--output_path /path/to/output_folder
```
Input:
- One Parquet file that contains the cell embeddings and metadata.

Parameters:
- `--input_path`: Path to Parquet file containing cell embeddings and metadata.
- `--output_path`: Path to folder where the ROC plot will be directed to.
- `--on_channel`: Specify channel to predict class labels on.
- `--intensity_thresh`: Specify the intensity threshold to binarise intensity values (from Channel 1) on.
- `--sample_confirm`: (Optional) Specify the channel for which the ground truth labels, predicted classes, and predicted probabilities will be saved.

Output:
- Train loss, Valid loss, and F1 score per epoch will be shown on the terminal.
- Confusion matrix will be returned at the end, on the terminal.
- One PNG file containing the ROC plot.
- One CSV file containing the ground truth labels, predicted classes, and predicted probabilities for the points from the specified sample/all samples in the test set, along with their coordinates.