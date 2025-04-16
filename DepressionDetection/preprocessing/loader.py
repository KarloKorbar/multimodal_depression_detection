import os
import pandas as pd
from typing import List, Optional


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
