# ## Exploratory Data Analysis
#
# Analyze the distribution of features and their relationships across modalities.
# TODO: do i even need EDA for the multimodal stuff sice i already did it all in the individual modalities?
def analyze_feature_distributions(df: pd.DataFrame) -> None:
    """Analyze distributions of features from different modalities.

    Args:
        df: DataFrame containing all features
    """
    # Group features by modality
    text_features = [col for col in df.columns if 'TEXT_' in col]
    audio_features = [col for col in df.columns if 'AUDIO_' in col or 'FORMANT_' in col or 'COVAREP_' in col]
    face_features = [col for col in df.columns if 'CLNF' in col]

    # Plot distributions for each modality
    for modality, features in [('Text', text_features), ('Audio', audio_features), ('Face', face_features)]:
        if not features:
            continue

        plt.figure(figsize=FIGURE_SIZE)
        df[features].boxplot()
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{modality} Features Distribution')
        plt.tight_layout()
        plt.show()


def analyze_correlations(df: pd.DataFrame) -> None:
    """Analyze correlations between features from different modalities.

    Args:
        df: DataFrame containing all features
    """
    # Group features by modality
    text_features = [col for col in df.columns if 'TEXT_' in col]
    audio_features = [col for col in df.columns if 'AUDIO_' in col or 'FORMANT_' in col or 'COVAREP_' in col]
    face_features = [col for col in df.columns if 'CLNF' in col]

    # Calculate and plot correlations between modalities
    for modality1, features1 in [('Text', text_features), ('Audio', audio_features), ('Face', face_features)]:
        for modality2, features2 in [('Text', text_features), ('Audio', audio_features), ('Face', face_features)]:
            if modality1 >= modality2 or not features1 or not features2:
                continue

            plt.figure(figsize=FIGURE_SIZE)
            correlation = df[features1 + features2].corr()
            sns.heatmap(correlation, cmap='coolwarm', center=0)
            plt.title(f'Correlation between {modality1} and {modality2} Features')
            plt.tight_layout()
            plt.show()

# Perform EDA
analyze_feature_distributions(df)
analyze_correlations(df)
