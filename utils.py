import torch
import numpy as np

def format_number(number, sig_figs=6):
    """Format a number to a specified number of significant figures."""
    if number == 0:
        return "0"
    return f"{number:.{sig_figs}g}"

def analyze_tensor(tensor, sig_figs=6):
    """Used to get a feel for actvations

    Args:
        tensor (_type_): _description_
        sig_figs (int, optional): _description_. Defaults to 6.

    Raises:
        ValueError: _description_
    """
    # Ensure the tensor is 1D or 2D
    if tensor.dim() > 2:
        raise ValueError("This function only supports 1D or 2D tensors")
    
    # Convert to 64-bit precision
    tensor = tensor.to(torch.float64)
    
    # Flatten the tensor if it's 2D
    flat_tensor = tensor.flatten() if tensor.dim() == 2 else tensor
    
    # Basic statistics
    mean = torch.mean(flat_tensor)
    median = torch.median(flat_tensor)
    std_dev = torch.std(flat_tensor)
    min_val = torch.min(flat_tensor)
    max_val = torch.max(flat_tensor)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Mean: {format_number(mean.item(), sig_figs)}")
    print(f"Median: {format_number(median.item(), sig_figs)}")
    print(f"Standard Deviation: {format_number(std_dev.item(), sig_figs)}")
    print(f"Min: {format_number(min_val.item(), sig_figs)}")
    print(f"Max: {format_number(max_val.item(), sig_figs)}")
    
    # Sparsity
    zero_count = torch.sum(flat_tensor == 0)
    sparsity = zero_count.item() / flat_tensor.numel()
    print(f"Sparsity: {format_number(sparsity, sig_figs)} ({zero_count} zeros out of {flat_tensor.numel()} elements)")
    
    # Convert to numpy for further analysis (still in 64-bit precision)
    np_tensor = flat_tensor.numpy()
    
    # Outlier detection using IQR method
    Q1 = np.percentile(np_tensor, 25)
    Q3 = np.percentile(np_tensor, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = np.logical_or(np_tensor < lower_bound, np_tensor > upper_bound)
    outlier_percentage = np.sum(outliers) / len(np_tensor) * 100
    
    print(f"Outlier percentage: {format_number(outlier_percentage, sig_figs)}%")
    print(f"Outlier range: < {format_number(lower_bound, sig_figs)} or > {format_number(upper_bound, sig_figs)}")

    # Distribution characteristics
    skewness = torch.mean(((flat_tensor - mean) / std_dev) ** 3)
    kurtosis = torch.mean(((flat_tensor - mean) / std_dev) ** 4) - 3
    
    print(f"Skewness: {format_number(skewness.item(), sig_figs)}")
    print(f"Kurtosis: {format_number(kurtosis.item(), sig_figs)}")