import pandas as pd
import numpy as np
import librosa

def add_timestamps(df, sampling_rate=100):
    # Calculate time interval based on the sampling rate
    time_interval = 1 / sampling_rate

    # Generate timestamps based on the number of samples
    num_samples = len(df)
    timestamps = np.arange(0, num_samples * time_interval, time_interval)

    # Add the timestamps as a new column in the DataFrame
    df['timestamp'] = timestamps[:num_samples]

    return df


def preprocess_AUDIO(path):
    # this is incorrect, i need to turn this into a dataframe with correctly labled shitt
    y, sr = librosa.load(path, sr=None)

    # Extract various features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Aggregate the features
    mfccs_mean = pd.DataFrame(np.mean(mfccs, axis=1)).add_prefix('MFCCSmean_')
    chroma_mean = pd.DataFrame(np.mean(chroma, axis=1)).add_prefix('CHROMAmean_')
    zcr_mean = pd.DataFrame(np.mean(zcr, axis=1)).add_prefix('ZCRmean_')

    audio_features = pd.DataFrame()

    audio_features = pd.concat([audio_features, mfccs_mean], axis=1)
    audio_features = pd.concat([audio_features, chroma_mean], axis=1)
    audio_features = pd.concat([audio_features, zcr_mean], axis=1)

    return audio_features


def preprocess_FORMANT(path):
    formant_features = pd.read_csv(path, header=None)
    FORMANT_FEATURES = ['F1', 'F2', 'F3', 'F4', 'F5']
    formant_features.columns = FORMANT_FEATURES
    formant_features = add_timestamps(formant_features)
    return formant_features


def preprocess_COVAREP(path):
    covarep_features = pd.read_csv(path, header=None)
    COVAREP_FEATURES = [
        'F0',
        'VUV',
        'NAQ',
        'QOQ',
        'H1H2',
        'PSP',
        'MDQ',
        'peakSlope',
        'Rd',
        'Rd_conf',
        'creak',

        'MCEP_0',
        'MCEP_1',
        'MCEP_2',
        'MCEP_3',
        'MCEP_4',
        'MCEP_5',
        'MCEP_6',
        'MCEP_7',
        'MCEP_8',
        'MCEP_9',
        'MCEP_10',
        'MCEP_11',
        'MCEP_12',
        'MCEP_13',
        'MCEP_14',
        'MCEP_15',
        'MCEP_16',
        'MCEP_17',
        'MCEP_18',
        'MCEP_19',
        'MCEP_20',
        'MCEP_21',
        'MCEP_22',
        'MCEP_23',
        'MCEP_24',

        'HMPDM_0',
        'HMPDM_1',
        'HMPDM_2',
        'HMPDM_3',
        'HMPDM_4',
        'HMPDM_5',
        'HMPDM_6',
        'HMPDM_7',
        'HMPDM_8',
        'HMPDM_9',
        'HMPDM_10',
        'HMPDM_11',
        'HMPDM_12',
        'HMPDM_13',
        'HMPDM_14',
        'HMPDM_15',
        'HMPDM_16',
        'HMPDM_17',
        'HMPDM_18',
        'HMPDM_19',
        'HMPDM_20',
        'HMPDM_21',
        'HMPDM_22',
        'HMPDM_23',
        'HMPDM_24',

        'HMPDD_0',
        'HMPDD_1',
        'HMPDD_2',
        'HMPDD_3',
        'HMPDD_4',
        'HMPDD_5',
        'HMPDD_6',
        'HMPDD_7',
        'HMPDD_8',
        'HMPDD_9',
        'HMPDD_10',
        'HMPDD_11',
        'HMPDD_12',
    ]
    covarep_features.columns = COVAREP_FEATURES
    covarep_features = add_timestamps(covarep_features)
    return covarep_features