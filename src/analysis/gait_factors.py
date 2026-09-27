"""Varimax-rotated PCA of gait-normalized muscle activity."""
import numpy as np
from sklearn.decomposition import PCA


def bartlett_sphericity(X):
    from scipy import stats

    n, p = X.shape
    R = np.corrcoef(X.T)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        return np.nan, np.nan
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    df = p * (p - 1) / 2
    p_val = stats.chi2.sf(chi2, df)
    return float(chi2), float(p_val)

def kmo_measure(X):
    R = np.corrcoef(X.T)
    R_inv = np.linalg.pinv(R)
    D = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
    P = -D @ R_inv @ D
    np.fill_diagonal(P, 1.0)
    r2 = R ** 2
    p2 = P ** 2
    np.fill_diagonal(r2, 0.0)
    np.fill_diagonal(p2, 0.0)
    kmo = r2.sum() / (r2.sum() + p2.sum())
    return float(kmo)

def standardize_for_pca(A):
    row_max = A.max(axis=1, keepdims=True)
    row_max = np.where(row_max < 1e-12, 1.0, row_max)
    A_norm = A / row_max

    row_std = A_norm.std(axis=1, ddof=1, keepdims=True)
    row_std = np.where(row_std < 1e-12, 1.0, row_std)
    A_std = A_norm / row_std
    return A_std, A_norm, row_max, row_std

def varimax_rotation(loadings, max_iter=1000, tol=1e-6):
    n_vars, n_factors = loadings.shape
    if n_factors == 1:
        return loadings.copy(), np.eye(1)

    h2 = np.sqrt((loadings ** 2).sum(axis=1, keepdims=True))
    h2 = np.where(h2 < 1e-12, 1.0, h2)
    L = loadings / h2

    rotation = np.eye(n_factors)
    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                Lij = L[:, [i, j]]
                A = Lij[:, 0]
                B = Lij[:, 1]
                u = A ** 2 - B ** 2
                v = 2 * A * B
                p = u.sum()
                q = v.sum()
                r = (u ** 2 - v ** 2).sum()
                s = (2 * u * v).sum()
                num = 2 * (n_vars * s - p * q)
                den = n_vars * r - (p ** 2 - q ** 2)
                if abs(den) < 1e-12:
                    continue
                theta = np.arctan2(num, den) / 4
                c, s_ = np.cos(theta), np.sin(theta)
                R2 = np.array([[c, -s_], [s_, c]])
                L[:, [i, j]] = L[:, [i, j]] @ R2
                rotation[:, [i, j]] = rotation[:, [i, j]] @ R2
        if np.max(np.abs(rotation - old_rotation)) < tol:
            break

    rotated_loadings = L * h2
    return rotated_loadings, rotation

def run_varimax_pca(A, n_components_init=10,
                    eigenvalue_threshold=0.5):
    n_muscles, n_time = A.shape

    A_std, A_norm, row_max, row_std = standardize_for_pca(A)

    pca = PCA(n_components=min(n_components_init, min(n_muscles, n_time)))
    scores_T = pca.fit_transform(A_std.T)
    scores = scores_T.T
    loadings = pca.components_.T

    eigvals = pca.explained_variance_

    keep_mask = eigvals >= eigenvalue_threshold
    n_factors = keep_mask.sum()
    if n_factors == 0:
        raise ValueError("No eigenvalues meet the retention threshold")

    scores_keep  = scores[keep_mask, :]

    loadings_keep = loadings[:, keep_mask] * np.sqrt(eigvals[keep_mask])

    print(f"[pca] eigenvalues: {eigvals[:n_factors + 2].round(3)}")
    print(f"[pca] retained {n_factors} factors (eigenvalue ≥ {eigenvalue_threshold})")
    print(f"[pca] cumulative variance: "
          f"{pca.explained_variance_ratio_[:n_factors].cumsum().round(3)}")

    rotated_loadings, rotation_matrix = varimax_rotation(loadings_keep)

    scale = np.sqrt(eigvals[keep_mask])[:, None]
    rotated_factors = rotation_matrix.T @ (scores_keep / scale)


    return rotated_factors, rotated_loadings, n_factors, pca, rotation_matrix
