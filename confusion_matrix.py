from scipy.stats import norm
import numpy as np
import pandas as pdf

import numpy as np
def assign_labels2(ckov_recon, ckov_theoretical):
    labels = np.full(ckov_recon.shape[0], -1, dtype=int)  # Initialize labels array with -1
    threshold = np.arccos(1 / (1.2904 + 0.056 * 3))  # Calculate threshold

    for i in range(ckov_recon.shape[0]):
        recon_value = ckov_recon[i, 0]  # Extract the ckov recon value

        # Check for special conditions first
        if recon_value <= 0 or recon_value > threshold:
            continue  # leave as default value -1

        # Calculate the absolute differences with theoretical values
        differences = np.abs(ckov_theoretical[i, :] - recon_value)

        # Find the index (label) of the smallest difference
        labels[i] = np.argmin(differences)

    return labels

def assign_labels(ckov_recon, ckov_theoretical):
    labels = np.zeros(ckov_recon.shape[0], dtype=int)  # Initialize labels array
    threshold = np.arccos(1 / (1.2904 + 0.0056 * 3))  # Calculate threshold

    for i in range(ckov_recon.shape[0]):
        recon_value = ckov_recon[i, 0]  # Extract the ckov recon value

        # Check for special conditions first
        if recon_value <= 0 or recon_value > threshold:
            labels[i] = 3
            continue

        # Calculate the absolute differences with theoretical values
        differences = np.abs(ckov_theoretical[i, :] - recon_value)

        # Find the index (label) of the smallest difference
        labels[i] = np.argmin(differences)

    return labels
import numpy as np

def one_hot_encode_labels(labels, num_classes=3):
    # Initialize the one-hot encoded matrix
    one_hot_encoded = np.zeros((labels.shape[0], num_classes))

    for i, label in enumerate(labels):
        if label in [0, 1, 2]:  # Check if the label is 0, 1, or 2
            one_hot_encoded[i, label] = 1

    return one_hot_encoded

def compare_with_true(y_true, labels_encoded):
    # Assuming y_true is already one-hot encoded and has the same shape as labels_encoded
    correct_predictions = np.sum(np.all(y_true == labels_encoded, axis=1))
    total = labels_encoded.shape[0]

    accuracy = correct_predictions / total
    return accuracy, correct_predictions

# Assign labels based on the closest theoretical value or special condition
labels = assign_labels(train_ckov_recon, train_ckov_theoretical)

# One-hot encode the assigned labels
labels_encoded = one_hot_encode_labels(labels)

# Compare the one-hot encoded labels with the true labels
accuracy, correct_predictions = compare_with_true(y_train_true, labels_encoded)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Correct Predictions: {correct_predictions} out of {y_train_true.shape[0]}")

# # Example usage:
# # Assuming train_ckov_recon and train_ckov_theoretical are defined as per your shapes
# labels = assign_labels(train_ckov_recon, train_ckov_theoretical)

# labels[labels==3]


# train_ckov_recon[labels==3]
# # print(train_ckov_recon[train_ckov_recon>1])
# # train_ckov_recon[train_ckov_recon<0]
# # Example usage:
# # Assuming train_ckov_recon and train_ckov_theoretical are defined as per your shapes
# labels = assign_labels(train_ckov_recon, train_ckov_theoretical)

# labels[labels==3]


# train_ckov_recon[labels==3]
# # print(train_ckov_recon[train_ckov_recon>1])
# # train_ckov_recon[train_ckov_recon<0]

#cm = create_confusion_matrix(y_train_true, labels)

# fig, axs = plt.subplots(1, 4, figsize=(25, 6))

# # Assuming y_pred_train and y_pred_test are probabilities
# y_pred_train_labels = np.argmax(y_pred_train, axis=1)
# y_pred_test_labels = np.argmax(y_pred_test, axis=1)


# y_train_trueL = np.argmax(y_train_true, axis=1)
# y_test_trueL = np.argmax(y_test_true, axis=1)

# # Confusion Matrix for train data
# cm_train = confusion_matrix(y_train_trueL, y_pred_train_labels)
# plot_confusion_matrix(axs[2], cm_train, title="Train Confusion Matrix")


# Define the plotting function for the confusion matrix with an extra 'Rejected' category
def plot_confusion_matrix_htm(ax, cm, labels, title="Confusion Matrix"):
    """Utility function to plot the confusion matrix."""
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd' if cm.dtype.kind == 'i' else '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix_htm3(ax, cm, labels, title="Confusion Matrix"):
    """Utility function to plot the confusion matrix with percentages."""
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = (cm[i, j] / np.sum(cm[i])) * 100 if np.sum(cm[i]) > 0 else 0
            text = f"{cm[i, j]}\n({percent:.1f}%)"
            thresh = cm.max() / 2.
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return ax

def plot_confusion_matrix_htm2(ax, cm, title="Confusion Matrix"):
    """Utility function to plot the confusion matrix."""
    ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['Pion', 'Kaon', 'Proton'], fontsize=13)
    ax.set_yticklabels(['Pion', 'Kaon', 'Proton'], fontsize=13)
    ax.set_title(title, fontsize=LARGE_FONT_SIZE * 1.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = (cm[i, j] / np.sum(cm[i])) * 100 if np.sum(cm[i]) > 0 else 0
            text = f"{cm[i, j]}\n({percent:.1f}%)"
            thresh = cm.max() / 2.
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Assign labels based on the closest theoretical value or special condition
labels = assign_labels(train_ckov_recon, train_ckov_theoretical)


import numpy as np
def assign_labels2(ckov_recon, ckov_theoretical):
    labels = np.full(ckov_recon.shape[0], -1, dtype=int)  # Initialize labels array with -1
    threshold = np.arccos(1 / (1.2904 + 0.056 * 3))  # Calculate threshold

    for i in range(ckov_recon.shape[0]):
        recon_value = ckov_recon[i, 0]  # Extract the ckov recon value

        # Check for special conditions first
        if recon_value <= 0 or recon_value > threshold:
            continue  # Leave the label as -1 and move on

        # Calculate the absolute differences with theoretical values
        differences = np.abs(ckov_theoretical[i, :] - recon_value)

        # Find the index (label) of the smallest difference
        labels[i] = np.argmin(differences)

    return labels

def assign_labels(ckov_recon, ckov_theoretical):
    labels = np.zeros(ckov_recon.shape[0], dtype=int)  # Initialize labels array
    threshold = np.arccos(1 / (1.2904 + 0.0056 * 3))  # Calculate threshold

    for i in range(ckov_recon.shape[0]):
        recon_value = ckov_recon[i, 0]  # Extract the ckov recon value

        # Check for special conditions first
        if recon_value <= 0 or recon_value > threshold:
            labels[i] = 3
            continue

        # Calculate the absolute differences with theoretical values
        differences = np.abs(ckov_theoretical[i, :] - recon_value)

        # Find the index (label) of the smallest difference
        labels[i] = np.argmin(differences)

    return labels
import numpy as np

def one_hot_encode_labels(labels, num_classes=3):
    # Initialize the one-hot encoded matrix
    one_hot_encoded = np.zeros((labels.shape[0], num_classes))

    for i, label in enumerate(labels):
        if label in [0, 1, 2]:  # Check if the label is 0, 1, or 2
            one_hot_encoded[i, label] = 1

    return one_hot_encoded

def compare_with_true(y_true, labels_encoded):
    correct_predictions = np.sum(np.all(y_true == labels_encoded, axis=1))
    total = labels_encoded.shape[0]

    accuracy = correct_predictions / total
    return accuracy, correct_predictions

# Assign labels based on the closest theoretical value or special condition
labels = assign_labels(train_ckov_recon, train_ckov_theoretical)

# One-hot encode the assigned labels
labels_encoded = one_hot_encode_labels(labels)

# Compare the one-hot encoded labels with the true labels
accuracy, correct_predictions = compare_with_true(y_train_true, labels_encoded)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Correct Predictions: {correct_predictions} out of {y_train_true.shape[0]}")

# train_ckov_recon[labels==3]
# # print(train_ckov_recon[train_ckov_recon>1])
# # train_ckov_recon[train_ckov_recon<0]
# # Example usage:
# # Assuming train_ckov_recon and train_ckov_theoretical are defined as per your shapes
# labels = assign_labels(train_ckov_recon, train_ckov_theoretical)

# labels[labels==3]


# train_ckov_recon[labels==3]
# # print(train_ckov_recon[train_ckov_recon>1])
# # train_ckov_recon[train_ckov_recon<0]

#cm = create_confusion_matrix(y_train_true, labels)

# fig, axs = plt.subplots(1, 4, figsize=(25, 6))

# # Assuming y_pred_train and y_pred_test are probabilities
# y_pred_train_labels = np.argmax(y_pred_train, axis=1)
# y_pred_test_labels = np.argmax(y_pred_test, axis=1)


# y_train_trueL = np.argmax(y_train_true, axis=1)
# y_test_trueL = np.argmax(y_test_true, axis=1)

# # Confusion Matrix for train data
# cm_train = confusion_matrix(y_train_trueL, y_pred_train_labels)
# plot_confusion_matrix(axs[2], cm_train, title="Train Confusion Matrix")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
train_mask = (train_ckov_recon >= 0).reshape(-1)
test_mask = (test_ckov_recon >= 0).reshape(-1)

# Step 1: Assign labels for both training and testing data
print(train_ckov_recon.shape)
print(train_ckov_theoretical.shape)

train_ckov_recon_filt = train_ckov_recon[train_mask]
train_ckov_theoretical_filt = train_ckov_theoretical[train_mask]

test_ckov_recon_filt = test_ckov_recon[test_mask]
test_ckov_theoretical_filt = test_ckov_theoretical[test_mask]

print(f" test_ckov_recon_filt shape {test_ckov_recon_filt.shape}")
print(f" test_ckov_theoretical_filt shape {test_ckov_theoretical_filt.shape}")


train_labels = assign_labels(train_ckov_recon_filt, train_ckov_theoretical_filt)
test_labels = assign_labels(test_ckov_recon_filt, test_ckov_theoretical_filt)


y_train_true_filt =  y_train_true[train_mask]
y_test_true_filt = y_test_true[test_mask]


# Step 2: Convert y_train_true and y_test_true from one-hot encoded to class labels
y_train_true_labels = np.argmax(y_train_true_filt, axis=1)
y_test_true_labels = np.argmax(y_test_true_filt, axis=1)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(25, 6))


print(f" y_train_true_filt shape {y_train_true_filt.shape}")
print(f" y_test_true_filt shape {y_test_true_filt.shape}")
print(f" y_train_true shape {y_train_true.shape}")
print(f" y_test_true shape {y_test_true.shape}")
print(f" y_train_true_labels shape {y_train_true_labels.shape}")
print(f" y_test_true_labels shape {y_test_true_labels.shape}")


valid_train_indices = ~np.isnan(train_labels)
valid_test_indices = ~np.isnan(test_labels)
categories = ['Pion', 'Kaon', 'Proton']  # Add 'Rejected' as the fourth category

# Confusion Matrix for training data
cm_train = confusion_matrix(y_train_true_labels[valid_train_indices], train_labels[valid_train_indices], labels=[0, 1, 2])

plot_confusion_matrix_htm3(axs[0], cm_train, categories, title="Train Confusion Matrix for HTM")

# Confusion Matrix for testing data

cm_test = confusion_matrix(y_test_true_labels[valid_test_indices], test_labels[valid_test_indices], labels=[0, 1, 2])
plot_confusion_matrix_htm3(axs[1], cm_test, categories, title="Test Confusion Matrix for HTM")

plt.tight_layout()
plt.show()
