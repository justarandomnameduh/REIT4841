import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix, classification_report, cohen_kappa_score

# Path to files
predictions_file = '/home/nqmtien/REIT4841/rag_cbm/predictions_gemma_4b.csv'
ground_truth_file = '/home/nqmtien/REIT4841/rag_cbm/isic_2018_task3/data/validation_ground_truth/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'

# Load data
predictions_df = pd.read_csv(predictions_file)
ground_truth_df = pd.read_csv(ground_truth_file)

# Merge on image ID to ensure we're comparing the same images
merged_df = predictions_df.merge(ground_truth_df, on='image')

# Extract class names
classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Prepare true and predicted labels
y_true = []
y_pred = []

# Extract and convert one-hot encoding to class indices
for idx, row in merged_df.iterrows():
    # Find true label (ground truth)
    true_label = None
    for category in classes:
        if row[f'{category}_y'] == 1.0:  # _y suffix for ground truth
            true_label = category
            break
    
    # Find predicted label
    pred_label = None
    for category in classes:
        if row[f'{category}_x'] == 1.0:  # _x suffix for predictions
            pred_label = category
            break
    
    y_true.append(true_label)
    y_pred.append(pred_label)

# Convert to numerical values for some metrics
class_to_idx = {cls: i for i, cls in enumerate(classes)}
y_true_idx = [class_to_idx[lbl] for lbl in y_true]
y_pred_idx = [class_to_idx[lbl] for lbl in y_pred]

# Convert to one-hot encoding for other metrics
def to_categorical(y, num_classes):
    result = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        result[i, label] = 1
    return result

y_true_onehot = to_categorical(y_true_idx, len(classes))
y_pred_onehot = to_categorical(y_pred_idx, len(classes))

# Calculate various metrics
print('ISIC 2018 Task 3 - Skin Lesion Classification Evaluation')
print('-' * 60)
print(f'Total samples: {len(y_true)}')

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'\nOverall Accuracy: {accuracy:.4f}')

# Balanced accuracy (accounts for class imbalance)
balanced_acc = balanced_accuracy_score(y_true_idx, y_pred_idx)
print(f'Balanced Accuracy: {balanced_acc:.4f}')

# Cohen's Kappa (agreement level)
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true_idx, y_pred_idx)
print(f'Matthews Correlation Coefficient: {mcc:.4f}')

# Class-wise metrics
print('\nClass-wise Performance:')
class_report = classification_report(y_true, y_pred, target_names=classes, digits=4)
print(class_report)

# Calculate metrics with different averaging strategies
print('\nMulti-class Metrics with Different Averaging Strategies:')
for avg in ['micro', 'macro', 'weighted']:
    precision = precision_score(y_true, y_pred, average=avg)
    recall = recall_score(y_true, y_pred, average=avg)
    f1 = f1_score(y_true, y_pred, average=avg)
    print(f'  {avg.capitalize()} averaging:')
    print(f'    Precision: {precision:.4f}')
    print(f'    Recall: {recall:.4f}')
    print(f'    F1-score: {f1:.4f}')

# Confusion matrix
cm = confusion_matrix(y_true_idx, y_pred_idx)
print('\nConfusion Matrix:')
print(cm)

# Class distribution in ground truth
print('\nClass Distribution in Ground Truth:')
for i, cls in enumerate(classes):
    count = sum(1 for label in y_true if label == cls)
    percentage = count / len(y_true) * 100
    print(f'  {cls}: {count} ({percentage:.2f}%)')

# Class-specific accuracy
print('\nClass-specific Accuracy:')
for i, cls in enumerate(classes):
    # Rows where true label is the current class
    cls_indices = [j for j, label in enumerate(y_true) if label == cls]
    if cls_indices:
        # Calculate accuracy for this class
        correct = sum(1 for j in cls_indices if y_pred[j] == y_true[j])
        cls_acc = correct / len(cls_indices)
        print(f'  {cls}: {cls_acc:.4f} ({correct}/{len(cls_indices)})')
    else:
        print(f'  {cls}: No samples')