import torch
import torch.nn as nn
import torch.optim as optim
import functools
from typing import List, Tuple, Callable, Any
from torch import Tensor

class BasisWeightModel(nn.Module):
    def __init__(self, num_basis_vectors: int, model_base: Any) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize weights with requires_grad=True by default since it's a Parameter
        self.weights = nn.Parameter(torch.randn(num_basis_vectors, device=device, dtype=torch.bfloat16))
        self.model_base = model_base
    
    def forward(
        self,
        basis_vectors: Tensor,
        get_refusal_scores_fn: Callable,
        get_pre_hook_fn: Callable,
        harmful_instructions: List[str],
        batch_size: int
    ) -> Tensor:
        # Ensure basis vectors have requires_grad=True
        basis_vectors = basis_vectors.to(device=self.weights.device, dtype=self.weights.dtype)
        basis_vectors.requires_grad_(True)
        
        # Combine basis vectors with learned weights
        weighted_direction = torch.sum(
            basis_vectors * self.weights.unsqueeze(1),
            dim=0,
            keepdim=False
        )
        
        # Normalize the weighted direction while maintaining gradient flow
        norm = torch.norm(weighted_direction, p=2)
        weighted_direction = weighted_direction / (norm + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Create hooks using the weighted direction
        fwd_pre_hooks: List[Tuple[Any, Callable]] = [
            (self.model_base.model_block_modules[layer], 
             get_pre_hook_fn(direction=weighted_direction))  # Remove detach() and clone()
            for layer in range(self.model_base.model.config.num_hidden_layers)
        ]
        fwd_hooks: List = []
        
        # Create partial function with the new hooks
        scores_fn = functools.partial(
            get_refusal_scores_fn,
            model=self.model_base.model,
            instructions=harmful_instructions,
            tokenize_instructions_fn=self.model_base.tokenize_instructions_fn,
            refusal_toks=self.model_base.refusal_toks,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size
        )
        
        # Get refusal scores
        refusal_scores = scores_fn()
        return refusal_scores.mean()

def get_direction_ablation_input_pre_hook(direction: Tensor) -> Callable:
    """
    Creates a pre-forward hook that projects out a component in the specified direction.
    
    Args:
        direction: The direction to project out, should be normalized
        
    Returns:
        Hook function that can be applied to model layers
    """
    def hook_fn(module: Any, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        activation = inputs[0]
        # Move direction to same device and dtype as activation, maintain requires_grad
        direction_device = direction.to(device=activation.device, dtype=activation.dtype)
        
        # Compute projection while maintaining gradient flow
        projection = (activation @ direction_device).unsqueeze(-1) * direction_device
        modified_activation = activation - projection
        
        return (modified_activation,) + inputs[1:]
        
    return hook_fn

def train_basis_weights(
    basis_vectors: Tensor,
    model_base: Any,
    harmful_instructions: List[str],
    get_refusal_scores_fn: Callable,
    batch_size: int = 32,
    num_epochs: int = 1000,
    lr: float = 0.01
) -> Tuple[BasisWeightModel, List[float]]:
    """
    Train weights to maximize mean refusal score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_vectors = basis_vectors.shape[0]
    
    # Initialize model and move to appropriate device
    model = BasisWeightModel(num_basis_vectors=num_vectors, model_base=model_base)
    model = model.to(device)
    
    # Convert basis vectors to bfloat16 and ensure requires_grad=True
    basis_vectors = basis_vectors.to(device=device, dtype=torch.bfloat16)
    basis_vectors.requires_grad_(True)
    
    # Enable gradient computation for the model
    model.train()
    
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    losses: List[float] = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Calculate mean refusal score with dynamically created hooks
        mean_score = model(
            basis_vectors=basis_vectors,
            get_refusal_scores_fn=get_refusal_scores_fn,
            get_pre_hook_fn=get_direction_ablation_input_pre_hook,
            harmful_instructions=harmful_instructions,
            batch_size=batch_size
        )
        
        # We want to maximize the score, so minimize negative
        loss = -mean_score
        
        # Ensure loss requires grad
        if not loss.requires_grad:
            raise RuntimeError("Loss does not require grad. Check if all operations maintain gradient information.")
            
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Weights: {model.weights.cpu().detach()}")
    
    return model, losses