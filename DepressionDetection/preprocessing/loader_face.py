import pandas as pd

from preprocessing.loader import DataLoader
from preprocessing.pp import pp_face as pp_face


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
