import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load and analyse the audio.wav file

# Load the audio file
audio_file = 'AUDIO.wav'
y, sr = librosa.load(audio_file, sr=None)

# Plot the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform of Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Extract fundamental frequencies (f0) using librosa's piptrack
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Plot fundamental frequencies
plt.figure(figsize=(10, 4))
plt.plot(f0, label='Fundamental Frequency (f0)')
plt.title('Fundamental Frequencies over Time')
plt.xlabel('Frames')
plt.ylabel('Frequency (Hz)')
plt.show()

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCCs')
plt.show()


# Load and analyse the formant.csv file

# Load the formant data
formant_data = pd.read_csv('FORMANT.csv')

# Display the first few rows
print(formant_data.head())

# Visualize formant frequencies (assuming columns 'F1', 'F2', 'F3' for formant data)
plt.figure(figsize=(10, 4))
plt.plot(formant_data['Time'], formant_data['F1'], label='F1')
plt.plot(formant_data['Time'], formant_data['F2'], label='F2')
plt.plot(formant_data['Time'], formant_data['F3'], label='F3')
plt.title('Formant Frequencies Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.show()


# Load and analyse the Covarep.csv file
# Load COVAREP data
covarep_data = pd.read_csv('COVAREP.csv')

# Display the first few rows
print(covarep_data.head())

# Plot pitch (assuming column 'Pitch' in the CSV file)
plt.figure(figsize=(10, 4))
plt.plot(covarep_data['Time'], covarep_data['Pitch'], label='Pitch')
plt.title('Pitch Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.show()

# Calculate and plot speed of speech (assuming a column 'SpeechRate')
plt.figure(figsize=(10, 4))
plt.plot(covarep_data['Time'], covarep_data['SpeechRate'], label='Speed of Speech')
plt.title('Speed of Speech Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speech Rate (syllables per second)')
plt.legend()
plt.show()


