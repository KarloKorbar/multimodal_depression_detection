import pandas as pd

from preprocessing.loader import DataLoader
from preprocessing.pp import pp_audio as pp_audio


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
