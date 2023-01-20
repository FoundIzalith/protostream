# Protostream 

This program learns to differentiate and identify languages by training itself on spoken language audio clips. [This Pytorch Implementation](https://github.com/hrshtv/pytorch-lmu) of the [Legendre Memory Unit](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf) is used for the machine learning core. 

# Usage

To use, Protostream requires a data folder containing audio clips and a metadata file containing labels for the data. Both should be placed in the same directly as the Protostream scripts. The default targets are 'data' and 'audio_data.csv' respectively. 

Alternative targets can be passed as input when running the program. Argument 1 (argv[1]) will be passed as the path to the data folder, and argument 2 (argv[2]) will be passed as the path to the metadata file. 

Protostream will not run if it cannot locate the data or metadata. 

Validation and training sets will be automatically split by the main script. 

# Requirements

Protostream runs on Python3 and uses the following packages:
- PyTorch
- Pandas
- SciPy 