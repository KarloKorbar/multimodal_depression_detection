base_directory = "data"
import os
import pandas as pd
import util.pp_text as pp_text
import util.pp_audio as pp_audio
import util.pp_face as pp_face


# Function to retrieve the necessary files
def get_path_map(base_dir=base_directory, required_files=[], folder_ids=None):
    # Create a list to store the data for the DataFrame
    data = []

    # Get the list of subfolders
    subfolders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    # If folder_ids is specified, filter the subfolders list based on the provided IDs
    if folder_ids:
        subfolders = [f for f in subfolders if int(f.split("_")[0]) in folder_ids]

    # Iterate through each (filtered) subfolder in the base directory
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)

        # Create a dictionary to store the file paths for the current subfolder
        formatted_subfolder = subfolder.split("_")[0]  # getting rid of _P
        subfolder_files = {"ID": formatted_subfolder}

        # Iterate through each required file
        for file_name in required_files:
            file_path = os.path.join(subfolder_path, f"{subfolder[:3]}_{file_name}")

            # Check if the file exists and add it to the dictionary
            formatted_file_name = file_name.split(".")[0]
            subfolder_files[formatted_file_name] = (
                file_path if os.path.exists(file_path) else None
            )

        # Append the dictionary to the data list
        data.append(subfolder_files)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    df["ID"] = df["ID"].astype(int)
    return df


def get_balanced_subset(percentage, random_state):
    df_all = get_path_map()
    df_all = append_PHQ_Binary(df_all)

    # Get the balanced subset
    # Calculate the desired size of the subset
    target_size = int(len(df_all) * percentage)

    # Split the dataframe by PHQ_Binary values
    df_0 = df_all[df_all["PHQ_Binary"] == 0]
    df_1 = df_all[df_all["PHQ_Binary"] == 1]

    # Determine the maximum number of samples for each PHQ_Binary group
    max_samples_per_class = min(len(df_0), len(df_1), target_size // 2)

    # Sample from each group
    sampled_0 = df_0.sample(n=max_samples_per_class, random_state=random_state)
    sampled_1 = df_1.sample(n=max_samples_per_class, random_state=random_state)

    # Combine the samples and shuffle
    balanced_subset = (
        pd.concat([sampled_0, sampled_1])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    df_subset = balanced_subset

    subset = df_subset["ID"].to_list()
    return subset


def append_PHQ_Binary(df):
    phq_paths = [
        "testing/dev_split_Depression_AVEC2017.csv",
        "testing/full_test_split.csv",
        "testing/train_split_Depression_AVEC2017.csv",
    ]

    phq_dataframes = []

    for path in phq_paths:
        phq_df = pd.read_csv(path)

        if "PHQ_Binary" in phq_df.columns:
            phq_column = "PHQ_Binary"
        elif "PHQ8_Binary" in phq_df.columns:
            phq_column = "PHQ8_Binary"
        else:
            continue

        phq_df = phq_df[["Participant_ID", phq_column]]
        phq_df.rename(columns={phq_column: "PHQ_Binary"}, inplace=True)
        phq_dataframes.append(phq_df)

    combined_phq_df = pd.concat(phq_dataframes, ignore_index=True)
    df = df.merge(combined_phq_df, how="left", left_on="ID", right_on="Participant_ID")

    df.drop(columns=["Participant_ID"], inplace=True)
    df.dropna(subset=["PHQ_Binary"], inplace=True)
    return df


def get_feature_subset_df(path_map):
    df = pd.DataFrame()
    feature_map = {
        "TRANSCRIPT": (pp_text.preprocess_TRANSCRIPT, "TRANSCRIPT_"),
        "AUDIO": (pp_audio.preprocess_AUDIO, "AUDIO_"),
        "FORMANT": (pp_audio.preprocess_FORMANT, "FORMANT_"),
        "COVAREP": (pp_audio.preprocess_COVAREP, "COVAREP_"),
        "CLNF_gaze": (pp_face.preprocess_CLNF_gaze, "CLNFgaze_"),
        "CLNF_AUs": (pp_face.preprocess_CLNF_AUs, "CLNFAUs_"),
        "CLNF_hog": (pp_face.preprocess_CLNF_hog, "CLNFhog_"),
        "CLNF_features": (pp_face.preprocess_CLNF_features, "CLNFfeatures_"),
        "CLNF_pose": (pp_face.preprocess_CLNF_pose, "CLNFpose_"),
        "CLNF_features3D": (pp_face.preprocess_CLNF_features3D, "CLNFfeatures3D_"),
    }

    for i in path_map.index:
        df_concat = pd.DataFrame()
        # Loop through feature_map and process if column exists in pathMap
        for column, (preprocess_func, prefix) in feature_map.items():
            if column in path_map.columns:
                processed_feature = preprocess_func(path_map[column][i]).add_prefix(
                    prefix
                )
                df_concat = pd.concat([df_concat, processed_feature], axis=1)
        # Add ID
        df_concat["ID"] = path_map["ID"].iloc[i]
        # Append the concatenated dataframe for each index to the main df
        df = pd.concat([df, df_concat], ignore_index=True)

    # Format column names properly
    df.columns = df.columns.str.replace(r"[^\w]", "", regex=True)

    return df


def get_binary(percentage=0, random_state=42):
    balanced_subset = get_balanced_subset(percentage, random_state)
    df = get_path_map(folder_ids=balanced_subset)
    df = append_PHQ_Binary(df)
    df.set_index("ID", inplace=True)

    return df


def get_text(percentage=0, random_state=42):
    text_features = [
        "TRANSCRIPT.csv",
    ]

    balanced_subset = get_balanced_subset(percentage, random_state)
    path_map = get_path_map(required_files=text_features, folder_ids=balanced_subset)
    df = get_feature_subset_df(path_map)
    df.set_index("ID", inplace=True)

    return df


def get_audio(percentage=0, random_state=42, ds_freq="50ms", rw_size="10s"):
    audio_features = [
        'AUDIO.wav',
        "FORMANT.csv",
        "COVAREP.csv",
    ]

    balanced_subset = get_balanced_subset(percentage, random_state)
    path_map = get_path_map(required_files=audio_features, folder_ids=balanced_subset)
    df = get_feature_subset_df(path_map)
    # Handle duplicate timestamp columns
    df["TIMESTAMP"] = df["FORMANT_timestamp"]
    df = df.drop(
        columns=[col for col in df.columns if "timestamp" in col and col != "TIMESTAMP"]
    )
    # Index & timestamp setup
    df["TIMESTAMP"] = pd.to_timedelta(df["TIMESTAMP"], unit="s")
    df.set_index(["ID", "TIMESTAMP"], inplace=True)
    # down-sampling the high frequency audio data to match the low frequency facial data for multi-modality
    df = df.groupby("ID").resample("33.3311ms", level="TIMESTAMP").mean()
    # Rounding TIMESTAMP for consistency between audio & face data
    df = df.reset_index()
    df["TIMESTAMP"] = df["TIMESTAMP"].apply(lambda x: x.round("10ms"))
    df.set_index(["ID", "TIMESTAMP"], inplace=True)

    # Down sampling the data
    df_resampled = df.groupby("ID").resample(ds_freq, level="TIMESTAMP").mean()
    # Applying a rolling window to smooth out the data
    df_smoothed = (
        df_resampled.groupby(level="ID")
        .rolling(rw_size, on=df_resampled.index.get_level_values("TIMESTAMP"))
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df_smoothed


def get_face(percentage=0, random_state=42, ds_freq="50ms", rw_size="10s"):
    face_features = [
        "CLNF_gaze.txt",
        "CLNF_AUs.txt",
        "CLNF_hog.bin",
        "CLNF_features.txt",
        "CLNF_pose.txt",
        "CLNF_features3D.txt",
    ]

    balanced_subset = get_balanced_subset(percentage, random_state)
    path_map = get_path_map(required_files=face_features, folder_ids=balanced_subset)
    df = get_feature_subset_df(path_map)
    # Get rid of unnecessary columns
    df = df.drop(
        columns=[
            col
            for col in df.columns
            if any(substring in col for substring in ["frame", "confidence", "success", "is_valid"]) # TODO: check if is_valid is needed
        ]
    )
    # Handle duplicate timestamp columns
    df["TIMESTAMP"] = df["CLNFgaze_timestamp"]
    df = df.drop(
        columns=[col for col in df.columns if "timestamp" in col and col != "TIMESTAMP"]
    )
    # Index & timestamp setup
    df["TIMESTAMP"] = pd.to_timedelta(df["TIMESTAMP"], unit="s")
    df.set_index(["ID", "TIMESTAMP"], inplace=True)
    # Rounding TIMESTAMP for consistency between audio & face data
    df = df.reset_index()
    df["TIMESTAMP"] = df["TIMESTAMP"].apply(lambda x: x.round("10ms"))
    df.set_index(["ID", "TIMESTAMP"], inplace=True)

    # Down sampling the data
    df_resampled = df.groupby("ID").resample(ds_freq, level="TIMESTAMP").mean()
    # Applying a rolling window to smooth out the data
    df_smoothed = (
        df_resampled.groupby(level="ID")
        .rolling(rw_size, on=df_resampled.index.get_level_values("TIMESTAMP"))
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df_smoothed
