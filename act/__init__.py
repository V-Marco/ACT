# Experimental
import torch
import os

device = torch.device(
    "cuda" if torch.cuda.is_available() and not os.environ.get("NOCUDA") else "cpu"
)
if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
torch.set_default_device(device.type)
