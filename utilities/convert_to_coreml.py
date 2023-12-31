import torch
import coremltools as ct
import pandas 
import random
from torch.utils.data import DataLoader
from langIdentifier import audioData, langIdentifierReLU

# Load saved model
torchModel = langIdentifierReLU()
torchModel.load_state_dict(torch.load("trained_model.pt"))

torchModel.eval()

#Random data 
loader = torch.rand(64, 2, 64, 860)

# Trace model flow
print("Model loaded; tracing flow...")
traced = torch.jit.trace(torchModel, loader)
output = traced(loader)

# Convert to Core ML
print("Finished tracing")
print("Converting model...")
convertedModel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=loader.shape)]
)

print("Saving model to .mlpackage ...")

convertedModel.save("stream.mlpackage")

print("All done!")