# hubconf.py

from netvlad import NetVLAD  # Import your model class

def my_model():
    """
    Example function to load the model.
    Args:
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional arguments for model initialization.
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = NetVLAD()
    return model
