import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.2):
        super(DNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(input_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc_1(x)
        out = self.fc_2(self.relu(out))
        out = self.fc_2(self.relu(out))
        out = self.fc_2(self.relu(out))
        out = self.fc_2(self.relu(out))
        out = self.fc_2(self.relu(out))
        out = self.fc_2(self.relu(out))
        out = self.fc_3(self.relu(out))
        out = self.dropout(out)
        return out

def save_results_to_file(accuracies, aucs, fold_results, fold_indices, file_path):
    with open(file_path, 'w') as output_file:

        output_file.write(f'Mean Accuracy: {np.mean(accuracies):.2f}\n')
        output_file.write(f'Mean AUC: {np.mean(aucs):.2f}\n\n')

        for i, ((accuracies, aucs), indices) in enumerate(zip(fold_results, fold_indices)):
            output_file.write(f'Repeat {i+1}:\n')
            output_file.write(f'  Fold Accuracies: {accuracies}\n')
            output_file.write(f'  Fold AUCs: {aucs}\n')
            output_file.write(f'  Fold Indices: {indices}\n\n')

    print(f'Results have been saved to {file_path}')
def run_cross_validation(X, y, n_repeats,output_file):

    # Parameters for cross-validation
    num_epochs = 100
    batch_size = 64
    hidden_size = 64
    output_size = 1
    learning_rate = 0.001
    input_size = X.shape[1]

    accuracies = []
    aucs = []
    fold_results = []
    fold_indices = []
    for repeats in range(n_repeats):
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_aucs = []
        fold_indices_list = []

        # Perform cross-validation
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {fold_idx + 1}")

            # Prepare data for current fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model = DNN(input_size, hidden_size, output_size)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            best_model_state = None

            # Training the model
            for epoch in range(num_epochs):
                model.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    model.eval()
                    best_model_state = model.state_dict().copy()
                    model.train()

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor)
                    y_pred = torch.sigmoid(y_pred)
                    y_pred_class = torch.round(y_pred)
                    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_class.numpy())
                    y_pred_decision = y_pred > 0.5
                    auc = roc_auc_score(y_test_tensor.numpy(), y_pred_decision.numpy())
                    fold_accuracies.append(accuracy)
                    fold_aucs.append(auc)
                    fold_indices_list.append(test_index.tolist())

        fold_results.append((fold_accuracies, fold_aucs))
        fold_indices.append(fold_indices_list)
        accuracies.append(np.mean(fold_accuracies))
        aucs.append(np.mean(fold_aucs))

        # Save results to file
    save_results_to_file(accuracies, aucs, fold_results, fold_indices, output_file)
def load_from_csv(X_filename='X.csv', y_filename='y.csv'):
    X = pd.read_csv(X_filename, header=None).values
    y = pd.read_csv(y_filename, header=None).values.ravel()
    return X, y

if __name__ == "__main__":
    X_mus, y_mus = load_from_csv('training_data/X_mus.csv', 'training_data/y_mus.csv')
    scaler = MinMaxScaler()
    X_mus = scaler.fit_transform(X_mus)
    output_file_mus = 'dnn_model_result_mus.txt'
    run_cross_validation(X_mus, y_mus, 1, output_file_mus)