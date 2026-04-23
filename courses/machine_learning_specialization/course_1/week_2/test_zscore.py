import numpy as np
from lab_utils_multi import zscore_normalize_features

# Test
X = np.c_[[0, 1, 2, 3, 4], [0, 1, 4, 9, 16], [0, 1, 8, 27, 64]]
print(f"X shape: {X.shape}")
print(f"X:\n{X}")

result = zscore_normalize_features(X)
print(f"\nResult type: {type(result)}")
print(f"Result is tuple: {isinstance(result, tuple)}")

if isinstance(result, tuple):
    X_norm, mu, sigma = result
    print(f"X_norm type: {type(X_norm)}, shape: {X_norm.shape}")
    print(f"mu type: {type(mu)}, shape: {mu.shape}")
    print(f"sigma type: {type(sigma)}, shape: {sigma.shape}")

    # Prueba np.ptp
    ptp_result = np.ptp(X_norm, axis=0)
    print(f"np.ptp(X_norm, axis=0): {ptp_result}")
else:
    print(f"ERROR: Expected tuple, got {type(result)}")
