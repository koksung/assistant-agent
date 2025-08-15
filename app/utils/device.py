import os
import torch

def configure_torch_device():
    try:
        if torch.cuda.is_available():
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError("No CUDA device available.")
    except Exception as e:
        print(f"⚠️ Falling back to CPU. Reason: {e}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
