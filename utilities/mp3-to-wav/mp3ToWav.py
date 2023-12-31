import sys
from os import path, walk 
from pydub import AudioSegment

#Convert whole folders of mp3 files into wav 

input_folder = sys.argv[1]
output_folder = sys.argv[2]

if not path.exists(input_folder):
    print("Error: No input folder")
    exit()
if not path.exists(output_folder):
    output_folder = "Output"
    os.makedirs(output_folder)

count = 0
for (root, dirnames, filenames) in walk(input_folder):                    
    for filename in filenames:
        if filename.endswith('.mp3'):
            newFileName = path.basename(filename).split('.')[0] + '.wav'
            original = input_folder + "/" + filename
            dest = output_folder + "/" + newFileName
            converted = AudioSegment.from_mp3(original)
            converted.export(dest, format="wav")
            count = count + 1

print("Converted ", count, " .mp3 files into .wav")

