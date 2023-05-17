import torch
import coremltools as ct
import pandas 
import sys
import random
from torch.utils.data import DataLoader
from langIdentifier import audioData, langIdentifierReLU

# Provide command line arguments here 
datapath = sys.argv[1] # Path to folder containing audio 
metadata = sys.argv[2] # Path to .csv containing metadata

# Load saved model
torchModel = langIdentifierReLU()
torchModel.load_state_dict(torch.load("trained_model.pt"))

torchModel.eval()

#Random data 
loader = torch.rand(64, 2, 64, 860)

# Trace model flow
print("Model loaded; tracing flow...")
traced = torch.jit.trace(torchModel, loader)
print("traced2")
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