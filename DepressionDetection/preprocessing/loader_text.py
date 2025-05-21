import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

from preprocessing.loader import DataLoader
from preprocessing.pp import pp_text as pp_text


class TextLoader(DataLoader):
    def __init__(self, base_directory: str = "data_input"):
        super().__init__(base_directory)
        self.feature_map = {
            "TRANSCRIPT": (pp_text.preprocess_TRANSCRIPT, "TRANSCRIPT_"),
        }
        self.required_files = ["TRANSCRIPT.csv"]

    def _preprocess_text(self, text: str) -> str:
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)

    def get_data(self, percentage: float = 0, random_state: int = 42) -> pd.DataFrame:
        balanced_subset = self.get_balanced_subset(percentage, random_state)
        path_map = self.get_path_map(
            required_files=self.required_files, folder_ids=balanced_subset
        )
        df = self._get_feature_subset_df(path_map)
        df["TRANSCRIPT_text"] = df["TRANSCRIPT_text"].apply(self._preprocess_text)
        return df.set_index("ID")
