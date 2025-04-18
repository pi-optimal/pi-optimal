import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pi_optimal.models.torch.base_torch_model import BaseTorchModel
# ------------------------------
# Helper function: get the best available device
# ------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
#    elif torch.backends.mps.is_available():
#        return torch.device("mps")
    else:
        return torch.device("cpu")

# ------------------------------
# Helper function: get activation module
# ------------------------------
def _get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'tanh':
        return nn.Tanh()
    elif activation_str in ['logistic', 'sigmoid']:
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {activation_str}")

# ------------------------------
# NeuralNetworkTorch class that mimics your repo structure.
# ------------------------------
class NeuralNetworkTorch(BaseTorchModel):
    def __init__(self, params: dict = {}):
        """
        Neural Network class that uses the underlying PyTorch MLPRegressorTorch or MLPClassifierTorch.
        """
        self.params = params.copy()
        self.use_past_states_for_reward = self.params.get("use_past_states_for_reward", True)
        self.params.pop("use_past_states_for_reward", None)
        
        self.models = []
        self.dataset_config = None

    def _create_estimator(self, state_configs, state_idx):
        """
        Create an estimator for the given state index and append it to self.models.
        """
        state_config = state_configs[state_idx]
        feature_type = state_config["type"]
        if feature_type == "numerical":
            model = MLPRegressorTorch(**self.params)
        elif feature_type in ["categorial", "binary"]:
            model = MLPClassifierTorch(**self.params)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        self.models.append(model)
        return model

# ------------------------------
# MLPRegressorTorch
# ------------------------------
class MLPRegressorTorch:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        power_t=0.5,  # not used (only relevant for learning_rate schedules)
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,  # only used if solver=='sgd'
        nesterovs_momentum=True,  # only used if solver=='sgd'
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000  # not used
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_str = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.device = get_device()  # Automatically detect device
        # These will be set during fit:
        self.model = None
        self.is_fitted = False
        self.input_dim = None
        self.output_dim = None

    def _build_model(self, input_dim, output_dim):
        """
        Build the network architecture as a torch.nn.Sequential.
        For regression, the final layer is linear without activation.
        """
        layers = []
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes)
        act_fn = _get_activation(self.activation_str)
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(act_fn)
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        model = nn.Sequential(*layers)
        return model.to(self.device)

    def fit(self, X, y):
        """
        Fit the MLP regressor.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) or (n_samples, n_outputs)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        n_samples, n_features = X.shape
        if y.ndim == 1:
            output_dim = 1
            y = y.reshape(-1, 1)
        else:
            output_dim = y.shape[1]
        self.input_dim = n_features
        self.output_dim = output_dim

        self.model = self._build_model(n_features, output_dim)

        bs = self.batch_size if self.batch_size != 'auto' else min(200, n_samples)
        # Ensure that tensors are created on the correct device later
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=self.shuffle)

        criterion = nn.MSELoss()
        if self.solver == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init,
                                   weight_decay=self.alpha,
                                   betas=(self.beta_1, self.beta_2),
                                   eps=self.epsilon)
        else:
            raise ValueError("Currently, only the 'adam' solver is supported in this implementation.")

        best_loss = np.inf
        epochs_no_improve = 0

        if self.early_stopping:
            split = int(n_samples * (1 - self.validation_fraction))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        else:
            train_loader = dataloader

        for epoch in range(self.max_iter):
            self.model.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {epoch_loss:.6f}")

            if self.early_stopping:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        out = self.model(xb)
                        val_loss = criterion(out, yb)
                        val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
                if self.verbose:
                    print(f"Validation Loss: {avg_val_loss:.6f}")
                if avg_val_loss + self.tol < best_loss:
                    best_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict values for X.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before prediction.")
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy()

# ------------------------------
# MLPClassifierTorch
# ------------------------------
class MLPClassifierTorch:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_str = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.device = get_device()  # Detect device automatically
        self.model = None
        self.is_fitted = False
        self.input_dim = None
        self.output_dim = None  # Number of classes

    def _build_model(self, input_dim, output_dim):
        """
        Build the network architecture as a torch.nn.Sequential.
        For classification, the final layer produces logits (no activation).
        """
        layers = []
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes)
        act_fn = _get_activation(self.activation_str)
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(act_fn)
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        model = nn.Sequential(*layers)
        return model.to(self.device)

    def fit(self, X, y):
        """
        Fit the MLP classifier.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples, ) with class labels
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.output_dim = len(classes)
        self.input_dim = n_features

        self.classes_ = classes
        label_map = {label: idx for idx, label in enumerate(classes)}
        y_indices = np.vectorize(label_map.get)(y)
        y_indices = y_indices.astype(np.int64)

        self.model = self._build_model(n_features, self.output_dim)

        bs = self.batch_size if self.batch_size != 'auto' else min(200, n_samples)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_indices))
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=self.shuffle)

        criterion = nn.CrossEntropyLoss()
        if self.solver == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init,
                                   weight_decay=self.alpha,
                                   betas=(self.beta_1, self.beta_2),
                                   eps=self.epsilon)
        else:
            raise ValueError("Currently, only the 'adam' solver is supported in this implementation.")

        best_loss = np.inf
        epochs_no_improve = 0

        if self.early_stopping:
            split = int(n_samples * (1 - self.validation_fraction))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y_indices[:split], y_indices[split:]
            train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=self.shuffle)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        else:
            train_loader = dataloader

        for epoch in range(self.max_iter):
            self.model.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {epoch_loss:.6f}")

            if self.early_stopping:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        out = self.model(xb)
                        val_loss = criterion(out, yb)
                        val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
                if self.verbose:
                    print(f"Validation Loss: {avg_val_loss:.6f}")
                if avg_val_loss + self.tol < best_loss:
                    best_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before prediction.")
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
        inv_label_map = {idx: label for idx, label in enumerate(self.classes_)}
        return np.array([[inv_label_map[p] for p in preds]])
