import pandas as pd
import numpy as np
import struct

def read_hog(filename, batch_size=5000):
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4))
        num_rows, = struct.unpack("i", f.read(4))
        num_channels, = struct.unpack("i", f.read(4))

        # The first four bytes encode a boolean value whether the frame is valid
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features))
        all_feature_vectors.append(feature_vector)

        # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        # Read in batches of given batch_size
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        # Multiply by 4 because of float32
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            # For comparison how many bytes were actually read
            num_bytes_read = len(bytes)
            if num_bytes_read == 0:
                break

            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector

            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            # Convert to array
            feature_vectors = np.array(feature_vectors).reshape(
                (num_feature_vectors_read, num_floats_per_feature_vector))
            # Discard the first three values in each row (num_cols, num_rows, num_channels)
            feature_vectors = feature_vectors[:, 3:]
            # Append to list of all feature vectors that have been read so far
            all_feature_vectors.append(feature_vectors)

            if num_bytes_read < num_bytes_to_read:
                break

        # Concatenate batches
        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)

        # Split into is-valid and feature vectors
        is_valid = all_feature_vectors[:, 0]
        hog_features = all_feature_vectors[:, 1:]

        # Create DataFrame
        df = pd.DataFrame({
            'is_valid': is_valid,
            'hog_features': list(hog_features)  # Store each row as a list in a DataFrame cell
        })

        return df

def preprocess_CLNF_gaze(path): # good
    gaze_df = pd.read_csv(path)
    return gaze_df

def preprocess_CLNF_AUs(path): # good
    au_df = pd.read_csv(path)
    return au_df

def preprocess_CLNF_hog(path): #TODO: 300 is .txt when it should be .bin
    df = read_hog(path)
    return df

def preprocess_CLNF_features(path):  # good
    features_df = pd.read_csv(path)
    return features_df

def preprocess_CLNF_pose(path):  # good
    pose_df = pd.read_csv(path)
    return pose_df

def preprocess_CLNF_features3D(path):  #good
    features3D_df = pd.read_csv(path)
    return features3D_df