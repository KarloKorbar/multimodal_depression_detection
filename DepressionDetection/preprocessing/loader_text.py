import pandas as pd

from preprocessing.loader import DataLoader
from preprocessing.pp import pp_text as pp_text


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
