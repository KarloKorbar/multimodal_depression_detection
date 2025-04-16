import pandas as pd

from preprocessing.loader import DataLoader


class ResultsLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {}

    def get_data(self, percentage: float = 0, random_state: int = 42) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        df = self.get_path_map(folder_ids=balanced_subset)
        df = self._append_PHQ_Binary(df)
        return df.set_index("ID")
