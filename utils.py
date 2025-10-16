import torch
from torch.utils.data import Dataset
import numpy as np

# =====================================
# DATA PREPARATION
# =====================================

class PrepareData(Dataset):
    """ Standard Dataset for X, y pairs. """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) if not torch.is_tensor(X) else X
        self.y = torch.from_numpy(y) if not torch.is_tensor(y) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PrepareData3D(Dataset):
    """ Dataset for X, y, z tuples, used by the convex solver. """
    def __init__(self, X, y, z):
        self.X = torch.from_numpy(X) if not torch.is_tensor(X) else X
        self.y = torch.from_numpy(y) if not torch.is_tensor(y) else y
        self.z = torch.from_numpy(z) if not torch.is_tensor(z) else z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx], self.z[idx].float()

def one_hot(labels, num_classes=10):
    """ Converts labels to one-hot encoding. """
    y = torch.eye(num_classes)
    return y[labels.long()]

# =====================================
# SIGN PATTERN GENERATION
# =====================================

def generate_sign_patterns(A, P, verbose=False):
    """ Generates P random sign patterns for the dataset A. """
    n, d = A.shape
    u_vectors = np.random.normal(0, 1, (d, P))
    sign_patterns_matrix = (np.matmul(A, u_vectors) >= 0)
    
    # Split the matrix into a list of pattern arrays
    sign_pattern_list = [sign_patterns_matrix[:, i] for i in range(P)]
    u_vector_list = [u_vectors[:, i] for i in range(P)]
    
    if verbose:
        print(f"Generated {len(sign_pattern_list)} sign patterns.")
    return len(sign_pattern_list), sign_pattern_list, u_vector_list

def generate_conv_sign_patterns(A_patches, P, verbose=False):
    """ Generates P convolutional sign patterns. """
    n, c, p1, p2 = A_patches.shape
    A = A_patches.reshape(n, -1)
    d = c * p1 * p2
    fs = 3  # 3x3 filter size
    
    sign_pattern_list = []
    u_vector_list = []
    
    for _ in range(P):
        # Create a random filter and apply it at a random location
        ind1 = np.random.randint(0, p1 - fs + 1)
        ind2 = np.random.randint(0, p2 - fs + 1)
        
        u_patch = np.zeros((c, p1, p2))
        u_patch[:, ind1:ind1 + fs, ind2:ind2 + fs] = np.random.normal(0, 1, (c, fs, fs))
        u_vector = u_patch.reshape(d, 1)
        
        sampled_sign_pattern = (np.matmul(A, u_vector) >= 0)[:, 0]
        sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(u_vector.flatten())
        
    if verbose:
        print(f"Generated {len(sign_pattern_list)} convolutional sign patterns.")
    return len(sign_pattern_list), sign_pattern_list, u_vector_list
