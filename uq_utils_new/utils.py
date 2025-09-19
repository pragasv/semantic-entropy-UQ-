from scipy.io import wavfile
import numpy as np
import os

def read_wav_file(file_path):
    sample_rate, data = wavfile.read(file_path)
    
    #downsample the data
    downsample_rate = 20
    downsampled_data = data[::downsample_rate]

    # Calculate the length of the middle 2000 samples
    middle_length = 2000
    start_index = len(downsampled_data) // 2 - middle_length // 2
    end_index = start_index + middle_length

    selected_samples = downsampled_data[start_index:end_index]

    return selected_samples

# Function to sort the .wav files in a folder and add them to the sequence
def add_files_to_sequence(file_paths_sequence, folder_path, signal_array):
    folder_path = os.path.join(folder_path, "audio")
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    wav_files.sort()  # Sort the files in alphabetical order (you can use your own sorting criteria)
    for i, file in enumerate(wav_files):
        if i >= 5:
            break 
        file_path = os.path.join(folder_path, file)
        file_paths_sequence.append(file_path)

        # read file 
        signal_array = np.vstack((signal_array, read_wav_file(file_path)))
    
    return file_paths_sequence, signal_array

def load_vidtmit_data(root_directory, signal_length):
    # Initialize an empty list to store the file paths in sequence
    file_paths_sequence = []
    signal_array = np.empty((0, signal_length))

    # Iterate through each subfolder and add the .wav files to the sequence
    subfolders = [f.name for f in os.scandir(root_directory) if f.is_dir()]
    subfolders.sort()  # Sort the subfolders in alphabetical order (you can use your own sorting criteria)
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_directory, subfolder)
        file_paths_sequence, signal_array = add_files_to_sequence(file_paths_sequence, subfolder_path, signal_array)
    
    return file_paths_sequence, signal_array
    