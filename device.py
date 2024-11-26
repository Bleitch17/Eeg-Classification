import torch


def get_system_device() -> torch.device:
    """
    Returns a PyTorch device object corresponding to the system GPU if available, otherwise CPU.
    """

    device = torch.device("cpu")

    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda:0")
        
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    elif torch.xpu.is_available():
        print("XPU is available")
        device = torch.device("xpu:0")

        print(f"Device name: {torch.xpu.get_device_name(0)}")
        print(f"Available XPU memory: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    else:
        print("No GPU available")

    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    # Test device availability
    get_system_device()
