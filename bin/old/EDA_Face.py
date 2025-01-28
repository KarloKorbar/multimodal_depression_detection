import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the data from text files
gaze_data = pd.read_csv('CLNF_gaze.txt', delim_whitespace=True)
au_data = pd.read_csv('CLNF_AUs.txt', delim_whitespace=True)
features_2d = pd.read_csv('CLNF_features.txt', delim_whitespace=True)
pose_data = pd.read_csv('CLNF_pose.txt', delim_whitespace=True)
features_3d = pd.read_csv('CLNF_features3D.txt', delim_whitespace=True)

# Reading binary file (HOG features)
def read_hog_bin_file(filename):
    with open(filename, 'rb') as f:
        n_cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        hog_features = np.fromfile(f, dtype=np.float32)
        hog_features = hog_features[1:].reshape(-1, n_cols)
    return hog_features

hog_features = read_hog_bin_file('CLNF_hog.bin')

# Analyzing gaze data
plt.figure(figsize=(10, 6))
sns.histplot(gaze_data, kde=True)
plt.title('Distribution of Gaze Data')
plt.show()

# Analyzing Action Units (AUs)
plt.figure(figsize=(12, 8))
sns.boxplot(data=au_data)
plt.title('Box Plot of Action Units (AUs)')
plt.xticks(rotation=90)
plt.show()

# Analyzing 2D facial landmarks
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features_2d.columns[::2], y=features_2d.columns[1::2], data=features_2d)
plt.title('2D Facial Landmarks')
plt.show()

# Analyzing 3D facial landmarks
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_3d.iloc[:, 0], features_3d.iloc[:, 1], features_3d.iloc[:, 2])
ax.set_title('3D Facial Landmarks')
plt.show()

# Analyzing head pose
plt.figure(figsize=(10, 6))
sns.lineplot(data=pose_data)
plt.title('Head Pose Analysis (Yaw, Pitch, Roll)')
plt.show()

# Performing PCA on HOG features to extract eigenfaces
pca = PCA(n_components=10)  # Number of principal components to keep
principal_components = pca.fit_transform(hog_features)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), explained_variance * 100)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance of Principal Components (Eigenfaces)')
plt.show()

# Visualizing the top eigenfaces
eigenfaces = pca.components_.reshape((10, int(np.sqrt(pca.components_.shape[1])), int(np.sqrt(pca.components_.shape[1]))))
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
plt.tight_layout()
plt.show()
