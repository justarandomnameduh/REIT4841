import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score

# Load the concept data from the JSON file
with open('skin_lesion_concept_abbreviations-4b.json', 'r') as f:
    concept_data = json.load(f)

# Define the specific concepts we want to use
selected_concepts = ['ESA', 'GP', 'PV', 'BDG', 'WLSA', 'SPC', 'PRL', 'PLF', 'APC']

# Create a dictionary mapping image_id to concept features
concept_features = {}
for item in concept_data:
    if 'image_id' in item and 'concepts' in item:
        image_id = item['image_id']
        # Initialize features with zeros
        features = {concept: 0 for concept in selected_concepts}
        # Set to 1 for concepts that are present
        for concept in item['concepts']:
            if concept in selected_concepts:
                features[concept] = 1
        concept_features[image_id] = features

# Load train and validation data
DATA_FOLDER = '../ham_concept_dataset/Datasets/metadata'
train_df = pd.read_csv(f'{DATA_FOLDER}/train.csv')
val_df = pd.read_csv(f'{DATA_FOLDER}/val.csv')

# Create feature matrices
def create_feature_df(df, concept_features):
    # Create DataFrame with zeros for all concepts
    feature_df = pd.DataFrame(0, index=df.index, columns=selected_concepts)
    
    # Fill in the features
    for idx, row in df.iterrows():
        image_id = row['image_id']
        if image_id in concept_features:
            for concept in selected_concepts:
                feature_df.loc[idx, concept] = concept_features[image_id][concept]
    
    return feature_df

# Create feature DataFrames
X_train = create_feature_df(train_df, concept_features)
y_train = train_df['benign_malignant']

X_val = create_feature_df(val_df, concept_features)
y_val = val_df['benign_malignant']
y_val_binary = (y_val == 1).astype(int)  # Treat 0.5 as benign (0)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.FloatTensor(y_val_binary.values)

# Define a standard linear model (without restricting to positive weights)
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # No ReLU constraint on weights
        return torch.sigmoid(self.linear(x))
    
    def get_coefficients(self):
        # Return the actual weights without ReLU
        return self.linear.weight.data.cpu().numpy()[0], self.linear.bias.data.cpu().numpy()[0]

# Create and train the model
model = LinearModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # No weight constraint enforcement needed
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get coefficients and bias
coefficients, intercept = model.get_coefficients()

# Display coefficients
coef_df = pd.DataFrame({
    'Feature': selected_concepts,
    'Coefficient': coefficients
})
print("\nFeature Coefficients (sorted by importance):")
print(coef_df.sort_values(by='Coefficient', ascending=False))

# Make predictions
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_val_tensor).squeeze().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_val_binary, y_pred)
report = classification_report(y_val_binary, y_pred)
conf_matrix = confusion_matrix(y_val_binary, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate additional metrics
auc = roc_auc_score(y_val_binary, y_pred_prob)
avg_precision = average_precision_score(y_val_binary, y_pred_prob)

print(f"\nROC AUC: {auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Display some sample predictions
sample_results = pd.DataFrame({
    'True Label': y_val.iloc[:10],
    'Predicted': y_pred[:10],
    'Probability': y_pred_prob[:10]
})

print("\nSample Predictions:")
print(sample_results)