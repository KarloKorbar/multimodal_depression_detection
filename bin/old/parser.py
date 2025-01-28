import os
import pandas as pd
# Define the directory containing the subfolders
base_directory = 'data'

# Define the list of required files
all_files = [
    'AUDIO.wav',
    'CLNF_gaze.txt',
    'FORMANT.csv',
    'CLNF_AUs.txt',
    'CLNF_hog.txt',
    'TRANSCRIPT.csv',
    'CLNF_features.txt',
    'CLNF_pose.txt',
    'CLNF_features3D.txt',
    'COVAREP.csv'
]

text_files = [
    'TRANSCRIPT.csv'
]
audio_files = [
    'AUDIO.wav',
    'FORMANT.csv',
    'COVAREP.csv'
]
face_files = [
    'CLNF_gaze.txt',
    'CLNF_AUs.txt',
    'CLNF_hog.txt',
    'CLNF_features.txt',
    'CLNF_pose.txt',
    'CLNF_features3D.txt',
]


# Function to retrieve the necessary files
def retrieve_files(base_dir, required_files):
    # Create a list to store the data for the DataFrame
    data = []

    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        # Ensure the path is a directory
        if os.path.isdir(subfolder_path):
            # Create a dictionary to store the file paths for the current subfolder
            subfolder_files = {'Subfolder': subfolder}

            # Iterate through each required file
            for file_name in required_files:
                file_path = os.path.join(subfolder_path, f"{subfolder[:3]}_{file_name}")

                # Check if the file exists and add it to the dictionary
                subfolder_files[file_name] = file_path if os.path.exists(file_path) else None

            # Append the dictionary to the data list
            data.append(subfolder_files)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df


# Retrieve the files and create the DataFrame
df = retrieve_files(base_directory, audio_files)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file for later use
df.to_csv('retrieved_files.csv', index=False)