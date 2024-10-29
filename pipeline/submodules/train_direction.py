import torch
import torch.nn as nn
import torch.optim as optim
import functools
from typing import List, Tuple, Callable, Any,Dict
from torch import Tensor

class BasisWeightModel(nn.Module):
    def __init__(self, num_basis_vectors: int, model_base: Any) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize weights with specified value
        self.weights = nn.Parameter(-torch.ones((num_basis_vectors), device=device, dtype=torch.bfloat16))
        self.model_base = model_base
    
    def compute_direction(self, basis_vectors: Tensor) -> Tensor:
        """
        Compute the weighted direction while preserving sign information.
        Scale the magnitude but maintain the overall direction.
        """
        # Combine basis vectors with learned weights
        weighted_direction = torch.sum(
            basis_vectors * self.weights.unsqueeze(1),
            dim=0,
            keepdim=False
        )
        
        # Scale the magnitude but preserve the sign
        # Using a softer normalization that preserves direction differences
        norm = torch.norm(weighted_direction, p=2)
        scale_factor = torch.log1p(norm) / (norm + 1e-6)
        weighted_direction = weighted_direction * scale_factor
        
        return weighted_direction
    
    def forward(
        self,
        basis_vectors: Tensor,
        get_refusal_scores_fn: Callable,
        get_pre_hook_fn: Callable,
        harmful_instructions: List[str],
        batch_size: int
    ) -> Tensor:
        basis_vectors = basis_vectors.to(device=self.weights.device, dtype=self.weights.dtype)
        basis_vectors.requires_grad_(True)
        
        # Use the new direction computation
        weighted_direction = self.compute_direction(basis_vectors)
        
        # Create hooks using the weighted direction
        fwd_pre_hooks: List[Tuple[Any, Callable]] = [
            (self.model_base.model_block_modules[layer], 
             get_pre_hook_fn(direction=weighted_direction))
            for layer in range(self.model_base.model.config.num_hidden_layers)
        ]
        fwd_hooks: List = []
        
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
    train_instructions: List[str],
    val_instructions: List[str],
    get_refusal_scores_fn: Callable,
    batch_size: int = 32,
    num_epochs: int = 1000,
    lr: float = 0.01,
    patience: int = 5,
    min_delta: float = 1e-10,
    weight_decay: float = 0.01,
) -> Tuple[BasisWeightModel, Dict[str, List[float]], Tensor]:
    """
    Train weights with sign-sensitive direction computation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_vectors = basis_vectors.shape[0]
    
    # Initialize model with specified initial weight value
    model = BasisWeightModel(num_vectors, model_base)
    model = model.to(device)
    
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience//2,
        verbose=True
    )
    
    basis_vectors = basis_vectors.to(device=device, dtype=torch.bfloat16)
    basis_vectors.requires_grad_(True)
    
    history = {
        'train_scores': [],
        'val_scores': [],
        'weights': [],
        'direction_norms': []
    }
    
    best_val_score = float('inf')
    best_model_state = None
    best_weighted_direction = None
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Store weights before update
        history['weights'].append(model.weights.detach().cpu())
        
        # Compute direction and its norm
        weighted_direction = model.compute_direction(basis_vectors)
        history['direction_norms'].append(torch.norm(weighted_direction).item())
        
        train_score = model(
            basis_vectors=basis_vectors,
            get_refusal_scores_fn=get_refusal_scores_fn,
            get_pre_hook_fn=get_direction_ablation_input_pre_hook,
            harmful_instructions=train_instructions,
            batch_size=batch_size
        )
        
        loss = train_score
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            weighted_direction = model.compute_direction(basis_vectors)
            val_score, _ = validate_basis_weights(
                model=model,
                weighted_direction=weighted_direction,
                validation_instructions=val_instructions,
                get_refusal_scores_fn=get_refusal_scores_fn,
                batch_size=batch_size
            )
        
        history['train_scores'].append(train_score.item())
        history['val_scores'].append(val_score)
        
        scheduler.step(val_score)
        
        if val_score < best_val_score - min_delta:
            best_val_score = val_score
            best_model_state = model.state_dict()
            best_weighted_direction = weighted_direction.clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Score: {train_score.item():.4f}, "
                  f"Val Score: {val_score:.4f}, "
                  f"Direction Norm: {history['direction_norms'][-1]:.4f}, "
                  f"Weights: {model.weights.detach().cpu()}")
        
        # if patience_counter >= patience:
        #     print(f"Early stopping triggered at epoch {epoch}")
        #     break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_weighted_direction

def validate_basis_weights(
    model: BasisWeightModel,
    weighted_direction: Tensor,
    validation_instructions: List[str],
    get_refusal_scores_fn: Callable,
    batch_size: int = 32
) -> Tuple[float, List[float]]:
    """
    Evaluate the trained basis weights model on a validation set of instructions.
    
    Args:
        model: Trained BasisWeightModel instance
        weighted_direction: Pre-computed weighted and normalized direction vector
        validation_instructions: List of instructions to validate against
        get_refusal_scores_fn: Function to compute refusal scores
        batch_size: Batch size for validation
        
    Returns:
        Tuple containing:
        - mean_validation_score: Average refusal score across all validation instructions
        - individual_scores: List of refusal scores for each instruction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move weighted direction to correct device and dtype
    weighted_direction = weighted_direction.to(device=device, dtype=torch.bfloat16)
    
    try:
        with torch.no_grad():
            # Create hooks using the pre-computed weighted direction
            fwd_pre_hooks = [
                (model.model_base.model_block_modules[layer],
                 get_direction_ablation_input_pre_hook(direction=weighted_direction))
                for layer in range(model.model_base.model.config.num_hidden_layers)
            ]
            
            # Get individual scores for each instruction
            individual_scores = get_refusal_scores_fn(
                model=model.model_base.model,
                instructions=validation_instructions,
                tokenize_instructions_fn=model.model_base.tokenize_instructions_fn,
                refusal_toks=model.model_base.refusal_toks,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=[],
                batch_size=batch_size
            ).tolist()
            
            # Calculate mean score
            mean_score = sum(individual_scores) / len(individual_scores)
            
            return mean_score, individual_scores
            
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise
    finally:
        # Return model to training mode if needed
        model.train()