# pi_optimal/models/pytorch/mlp_model.py

from pi_optimal.models.pytorch.pytorch_base_model import PytorchBaseModel

class MLPModel(PytorchBaseModel):
    """
    A PyTorch model with an MLP shared trunk.
    
    This model inherits from PytorchBaseModel and uses its default MLP implementation.
    """
    def __init__(self, dataset_config, shared_network_params=None, probabilistic=False, epochs=10, lr=1e-3, batch_size=32, device='cpu'):
        super().__init__(dataset_config, shared_network_params=None, probabilistic=probabilistic, epochs=epochs, lr=lr, batch_size=batch_size, device=device)
