import numpy as np
import joblib
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List
import os

def save_pca_components(pca: PCA, feature_names: List[str], output_path: str, 
                       explained_variance_threshold: float = 0.95) -> None:
    # Save PCA components and related information for later use.
    # Calculate number of components needed to reach threshold
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save PCA components and related information
    pca_info = {
        'components': pca.components_[:n_components],
        'mean': pca.mean_,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_components],
        'feature_names': feature_names,
        'n_components': n_components
    }
    
    joblib.dump(pca_info, output_path)

def load_and_transform_pca(input_data: np.ndarray, pca_path: str) -> Tuple[np.ndarray, Dict]:
    # Load saved PCA components and transform new data.
    pca_info = joblib.load(pca_path)
    
    # Center the data
    centered_data = input_data - pca_info['mean']
    
    # Transform data using saved components
    transformed_data = np.dot(centered_data, pca_info['components'].T)
    
    return transformed_data, pca_info

def get_important_features(pca_info: Dict) -> List[str]:
    # Get names of features that contribute most to the PCA components.
    # Get absolute component values
    abs_components = np.abs(pca_info['components'])
    
    # Calculate feature importance scores
    feature_importance = np.sum(abs_components, axis=0)
    
    # Get indices of most important features
    important_indices = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    # Return names of important features
    return [pca_info['feature_names'][i] for i in important_indices] 