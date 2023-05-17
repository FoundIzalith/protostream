# Protostream 

This program learns to differentiate and identify languages by training itself on spoken language audio clips. STREAM takes in .wav files and associated metadata, converts them into spectrograms, and loads them into a rectified linear unit machine learning model. Then, when provided with new audio inputs, it will attempt to identify the language spoken. 

## Usage

To use, Protostream requires a data folder containing audio clips and a metadata file containing labels for the data. Both should be placed in the same directly as the Protostream scripts. The default targets are 'data' and 'audio_data.csv' respectively. 

Alternative targets can be passed as input when running the program. Argument 1 (argv[1]) will be passed as the path to the data folder, and argument 2 (argv[2]) will be passed as the path to the metadata file. 

Protostream will not run if it cannot locate the data or metadata. 

Validation and training sets will be automatically split by the main script. 

## Requirements

Protostream runs on Python3 and uses the following packages:
- PyTorch
- Pandas
- SciPy 
- TQDM

