import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from preprocess import load_data, visualize_subjects
from tqdm import tqdm
import scipy.io as sio
import numpy as np
import time
from working.dim_reduce import apply_pca, apply_anova
from fusion import load_data_with_fusion

class SeedDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SeedMLP(nn.Module):
    def __init__(self, in_c = 100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_c, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.mlp(x)

def main():
    print("[Run Start]")
    print('[Load Data]')
    # all_data = load_data(is_dl=True)
    all_data = load_data_with_fusion()
    n_features = all_data.shape[-1]
    label = np.array([list(sio.loadmat('input/label.mat')['label'].squeeze(0)+1)*3 for _ in range(15)])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 30
    loo = LeaveOneOut()
    acc_results = []
    
    for train_idx, test_idx in loo.split(all_data):
        X_train, X_test = all_data[train_idx], all_data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        X_train, X_test = X_train.reshape(-1, n_features), X_test.reshape(-1, n_features)
        y_train, y_test = y_train.flatten(), y_test.flatten()

        # 归一化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # PCA|K=100
        # X_train, pca = apply_pca(X_train, n_components=100)
        # X_test = pca.transform(X_test)

        # ANOVA|K=160
        # X_train, anova = apply_anova(X_train, y_train, n_features=160)
        # X_test = anova.transform(X_test)

        train_dataset = SeedDataset(X_train, y_train)
        test_dataset = SeedDataset(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)
        model = SeedMLP(n_features).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(params=model.parameters(), lr=5e-4, weight_decay=1e-4)
        subject_acc = []
        for epoch in range(1, epochs+1):
            pbar1 = tqdm(train_dataloader, desc=f'MLP Training|[{epoch}/{epochs}]')
            print('\n[Train]')
            model.train()
            for X, y in pbar1:
                optimizer.zero_grad()
                pred = model(X.to(device))
                loss = criterion(pred, y.to(device))
                loss.backward()
                pbar1.set_postfix({"loss":loss.item()})
                optimizer.step()
            model.eval()
            print('[Test]')
            with torch.no_grad():
                pbar2 = tqdm(test_dataloader, desc=f'[MLP Testing]')
                correct = 0
                for X, y in pbar2:
                    pred = model(X.to(device))
                    loss = criterion(pred, y.to(device))
                    pbar2.set_postfix({"loss":loss.item()})
                    correct += (torch.argmax(pred, dim=1)==y.to(device)).sum().item()
            print(f"\n[Accuracy:{correct / len(test_dataset):.3f}]")
            if epochs - epoch <= 5: 
                subject_acc.append(correct / len(test_dataset))
        acc_results.append(np.mean(subject_acc))
    print('[Run Done]')

    return acc_results

if __name__ == '__main__':
    start = time.time()
    acc_results = main()
    end = time.time()
    print(f'MLP model evaluate mean score: {np.mean(acc_results):.3f}')
    visualize_subjects(acc_results, 'MLP_Fusion', 'MLP')
    print(f'Total time {end-start:.3f}')