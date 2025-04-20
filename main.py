import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
df = pd.read_excel(r'C:\\DewangData\\VSCode\\Endsem\\EndsemMOC\\ImmersiveCoolingExp_AllCycles.ods', engine='odf', sheet_name='Race Cycle')
df = df.dropna()

X = df.drop(columns=['Outlet fluid Temp. [°C]']).values
y = df['Outlet fluid Temp. [°C]'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create incidence matrix for hypergraph
def create_incidence_matrix(features, num_bins=5):
    num_nodes = features.shape[0]
    num_features = features.shape[1]
    hyperedges = []
    for feature_idx in range(num_features):
        feature = features[:, feature_idx]
        bins = np.linspace(feature.min(), feature.max(), num_bins + 1)
        for i in range(num_bins):
            members = np.where((feature >= bins[i]) & (feature < bins[i + 1]))[0]
            if len(members) > 1:
                hyperedges.append(members.tolist())
    num_edges = len(hyperedges)
    H = np.zeros((num_nodes, num_edges))
    for j, edge in enumerate(hyperedges):
        for node in edge:
            H[node, j] = 1.0
    return torch.tensor(H, dtype=torch.float32)

H = create_incidence_matrix(X_train.numpy())

# HGNN Layer
class HGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(HGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, H):
        Dv = torch.diag(torch.sum(H, dim=1))
        De = torch.diag(torch.sum(H, dim=0))
        Dv_inv_sqrt = torch.linalg.inv(torch.sqrt(Dv))
        De_inv = torch.linalg.inv(De)
        theta = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
        X_new = theta @ X
        return self.linear(X_new)

# HGNN Model
class HGNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(HGNNModel, self).__init__()
        self.hgnn1 = HGNNLayer(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.hgnn2 = HGNNLayer(hidden_features, out_features)

    def forward(self, X, H):
        out = self.hgnn1(X, H)
        out = self.relu(out)
        out = self.hgnn2(out, H)
        return out

# Define optimizers based on your list
def get_optimizer(name, params, lr=0.01):
    if name == "Adam":
        return optim.Adam(params, lr=lr)
    elif name == "AdaBelief":
        import adabelief_pytorch
        return adabelief_pytorch.AdaBelief(params, lr=lr)
    elif name == "AMSGrad":
        return optim.Adam(params, lr=lr, amsgrad=True)
    elif name == "SGD":
        return optim.SGD(params, lr=lr)  # Classic SGD
    elif name == "MomentumSGD":
        return optim.SGD(params, lr=lr, momentum=0.9)  # Momentum
    elif name == "RMSprop":
        return optim.RMSprop(params, lr=lr)
    elif name == "Adagrad":
        return optim.Adagrad(params, lr=lr)
    elif name == "Nadam":
        return optim.NAdam(params, lr=lr)
    elif name == "NAG":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif name == "Adadelta":
        return optim.Adadelta(params, lr=1.0)
    elif name == "Adamax":
        return optim.Adamax(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

optimizers_list = [
    "Adam", "AdaBelief", "AMSGrad", "SGD", "MomentumSGD",
    "RMSprop", "Adagrad", "Nadam", "NAG", "Adadelta", "Adamax"
]

results = {}

# Training loop
for opt_name in optimizers_list:
    print(f"\nTraining with {opt_name} optimizer")
    model = HGNNModel(X_train.shape[1], 64, 1)  # Hidden layer size 64
    try:
        optimizer = get_optimizer(opt_name, model.parameters())
    except Exception as e:
        print(f"Skipping {opt_name} due to error: {e}")
        continue

    criterion = nn.MSELoss()
    epochs = 200

    for epoch in range(epochs):
        model.train()
        output = model(X_train, H)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test phase
    H_test = create_incidence_matrix(X_test.numpy())
    model.eval()
    with torch.no_grad():
        predictions = model(X_test, H_test)

        y_test_np = y_test.numpy().flatten()
        predictions_np = predictions.numpy().flatten()

        mse = mean_squared_error(y_test_np, predictions_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, predictions_np)

        epsilon = 1e-10
        y_test_safe = np.where(y_test_np == 0, epsilon, y_test_np)
        mape = np.mean(np.abs((y_test_safe - predictions_np) / y_test_safe)) * 100
        msle = mean_squared_log_error(y_test_np, predictions_np)

    results[opt_name] = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MSLE": msle
    }

# Sort results by RMSE
sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])
sorted_data = []
for i, (opt_name, metrics) in enumerate(sorted_results[:5]):
    sorted_data.append({
        "Rank": i + 1,
        "Optimizer": opt_name,
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "MSLE": metrics["MSLE"]
    })

df_sorted_optimizers = pd.DataFrame(sorted_data)
print("\n=== Top 5 Optimizers ===")
print(df_sorted_optimizers.to_string(index=False))
