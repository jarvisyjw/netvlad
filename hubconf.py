# hubconf.py

from netvlad import NetVLAD  # Import your model class

def my_model(model_name='VGG16-NetVLAD-Pitts30K', whiten=True):
    """
    Example function to load the model.
    Args:
        pretrained (bool): If True, loads pretrained weights.
        **kwargs: Additional arguments for model initialization.
    Returns:
        torch.nn.Module: The loaded model.
    """
    conf = {"model_name": model_name, "whiten": whiten}
    model = NetVLAD(conf)
    return model
