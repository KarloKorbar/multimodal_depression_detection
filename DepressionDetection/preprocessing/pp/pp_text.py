import pandas as pd
import re


def preprocess_TRANSCRIPT(path):
    df = pd.read_csv(path, delimiter='\t')
    # Combine all rows of the 'value' column into one string
    transcript = ' '.join(df['value'].astype(str).tolist())
    # Clean the text
    transcript = re.sub(r'[^\w\s]', '', transcript.lower())
    # Wrap the result in a DataFrame
    result_df = pd.DataFrame({'text': [transcript]})
    return result_df
