import os
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
import preprocessing.pp_text as pp_text
import preprocessing.pp_audio as pp_audio
import preprocessing.pp_face as pp_face

#TODO: think about extracting all the models into their own files to make it cleaner

class DataLoader:
    def __init__(self, base_directory: str = "data_input"):
        self.base_directory = base_directory
        self.phq_paths = [
            "data_output/dev_split_Depression_AVEC2017.csv",
            "data_output/full_test_split.csv",
            "data_output/train_split_Depression_AVEC2017.csv",
        ]
        self.feature_map = {}

    def get_path_map(
        self, required_files: List[str] = [], folder_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        data = []
        subfolders = [
            f
            for f in os.listdir(self.base_directory)
            if os.path.isdir(os.path.join(self.base_directory, f))
        ]

        if folder_ids:
            subfolders = [f for f in subfolders if int(f.split("_")[0]) in folder_ids]

        for subfolder in subfolders:
            subfolder_path = os.path.join(self.base_directory, subfolder)
            formatted_subfolder = subfolder.split("_")[0]
            subfolder_files = {"ID": formatted_subfolder}

            for file_name in required_files:
                file_path = os.path.join(subfolder_path, f"{subfolder[:3]}_{file_name}")
                formatted_file_name = file_name.split(".")[0]
                subfolder_files[formatted_file_name] = (
                    file_path if os.path.exists(file_path) else None
                )

            data.append(subfolder_files)

        df = pd.DataFrame(data)
        df["ID"] = df["ID"].astype(int)
        return df

    def get_balanced_subset(self, percentage: float, random_state: int) -> List[int]:
        df_all = self.get_path_map()
        df_all = self._append_PHQ_Binary(df_all)

        target_size = int(len(df_all) * percentage)
        df_0 = df_all[df_all["PHQ_Binary"] == 0]
        df_1 = df_all[df_all["PHQ_Binary"] == 1]

        max_samples_per_class = min(len(df_0), len(df_1), target_size // 2)

        sampled_0 = df_0.sample(n=max_samples_per_class, random_state=random_state)
        sampled_1 = df_1.sample(n=max_samples_per_class, random_state=random_state)

        balanced_subset = (
            pd.concat([sampled_0, sampled_1])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

        return balanced_subset["ID"].to_list()

    def _append_PHQ_Binary(self, df: pd.DataFrame) -> pd.DataFrame:
        phq_dataframes = []

        for path in self.phq_paths:
            phq_df = pd.read_csv(path)
            phq_column = (
                "PHQ_Binary"
                if "PHQ_Binary" in phq_df.columns
                else "PHQ8_Binary" if "PHQ8_Binary" in phq_df.columns else None
            )

            if phq_column:
                phq_df = phq_df[["Participant_ID", phq_column]]
                phq_df.rename(columns={phq_column: "PHQ_Binary"}, inplace=True)
                phq_dataframes.append(phq_df)

        combined_phq_df = pd.concat(phq_dataframes, ignore_index=True)
        df = df.merge(
            combined_phq_df, how="left", left_on="ID", right_on="Participant_ID"
        )
        df.drop(columns=["Participant_ID"], inplace=True)
        return df.dropna(subset=["PHQ_Binary"])

    def _get_feature_subset_df(self, path_map: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame()

        for i in path_map.index:
            df_concat = pd.DataFrame()
            for column, (preprocess_func, prefix) in self.feature_map.items():
                if column in path_map.columns:
                    processed_feature = preprocess_func(path_map[column][i]).add_prefix(
                        prefix
                    )
                    df_concat = pd.concat([df_concat, processed_feature], axis=1)
            df_concat["ID"] = path_map["ID"].iloc[i]
            df = pd.concat([df, df_concat], ignore_index=True)

        df.columns = df.columns.str.replace(r"[^\w]", "", regex=True)
        return df

    def _process_temporal_features(
        self, df: pd.DataFrame, timestamp_col: str, ds_freq: str, rw_size: str
    ) -> pd.DataFrame:
        df["TIMESTAMP"] = df[timestamp_col]
        df = df.drop(
            columns=[
                col for col in df.columns if "timestamp" in col and col != "TIMESTAMP"
            ]
        )

        df["TIMESTAMP"] = pd.to_timedelta(df["TIMESTAMP"], unit="s")
        df.set_index(["ID", "TIMESTAMP"], inplace=True)

        if "AUDIO" in timestamp_col:
            df = df.groupby("ID").resample("33.3311ms", level="TIMESTAMP").mean()

        df = df.reset_index()
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(lambda x: x.round("10ms"))
        df.set_index(["ID", "TIMESTAMP"], inplace=True)

        df_resampled = df.groupby("ID").resample(ds_freq, level="TIMESTAMP").mean()
        return (
            df_resampled.groupby(level="ID")
            .rolling(rw_size, on=df_resampled.index.get_level_values("TIMESTAMP"))
            .mean()
            .reset_index(level=0, drop=True)
        )


class ResultsLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {}

    def get_data(self, percentage: float = 0, random_state: int = 42) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        df = self.get_path_map(folder_ids=balanced_subset)
        df = self._append_PHQ_Binary(df)
        return df.set_index("ID")


class TextLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {
            "TRANSCRIPT": (pp_text.preprocess_TRANSCRIPT, "TRANSCRIPT_"),
        }
        self.required_files = ["TRANSCRIPT.csv"]

    def get_data(self, percentage: float = 0, random_state: int = 42) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        path_map = self.get_path_map(
            required_files=self.required_files, folder_ids=balanced_subset
        )
        df = self._get_feature_subset_df(path_map)
        return df.set_index("ID")


class AudioLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {
            "AUDIO": (pp_audio.preprocess_AUDIO, "AUDIO_"),
            "FORMANT": (pp_audio.preprocess_FORMANT, "FORMANT_"),
            "COVAREP": (pp_audio.preprocess_COVAREP, "COVAREP_"),
        }
        self.required_files = ["AUDIO.wav", "FORMANT.csv", "COVAREP.csv"]

    def get_data(
        self,
        percentage: float = 0,
        random_state: int = 42,
        ds_freq: str = "50ms",
        rw_size: str = "10s",
    ) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        path_map = self.get_path_map(
            required_files=self.required_files, folder_ids=balanced_subset
        )
        df = self._get_feature_subset_df(path_map)
        return self._process_temporal_features(
            df, "FORMANT_timestamp", ds_freq, rw_size
        )


class FaceLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {
            "CLNF_gaze": (pp_face.preprocess_CLNF_gaze, "CLNFgaze_"),
            "CLNF_AUs": (pp_face.preprocess_CLNF_AUs, "CLNFAUs_"),
            "CLNF_hog": (pp_face.preprocess_CLNF_hog, "CLNFhog_"),
            "CLNF_features": (pp_face.preprocess_CLNF_features, "CLNFfeatures_"),
            "CLNF_pose": (pp_face.preprocess_CLNF_pose, "CLNFpose_"),
            "CLNF_features3D": (pp_face.preprocess_CLNF_features3D, "CLNFfeatures3D_"),
        }
        self.required_files = [
            "CLNF_gaze.txt",
            "CLNF_AUs.txt",
            "CLNF_hog.bin",
            "CLNF_features.txt",
            "CLNF_pose.txt",
            "CLNF_features3D.txt",
        ]

    def get_data(
        self,
        percentage: float = 0,
        random_state: int = 42,
        ds_freq: str = "50ms",
        rw_size: str = "10s",
    ) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        path_map = self.get_path_map(
            required_files=self.required_files, folder_ids=balanced_subset
        )
        df = self._get_feature_subset_df(path_map)
        df = self._process_temporal_features(df, "CLNFgaze_timestamp", ds_freq, rw_size)
        return df

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=[
                col
                for col in df.columns
                if any(
                    substring in col
                    for substring in ["frame", "confidence", "success", "is_valid"]
                )
            ]
        )
