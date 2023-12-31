

import torch
from torch import nn
import torchaudio
from torch.utils.mobile_optimizer import optimize_for_mobile

def get_demo_wrapper():

    wrapper = nn.ReLU()
    stream = torch.load("stream_model.pt")
    wrapper.load_state_dict(stream, strict=False)

    return wrapper

 

wrapper = get_demo_wrapper()

wrapper.eval()

 

scripted_model = torch.jit.script(wrapper)

optimized_model = optimize_for_mobile(scripted_model)

optimized_model._save_for_lite_interpreter("stream_asrv2.ptl")

print("Done _save_for_lite_interpreter")


