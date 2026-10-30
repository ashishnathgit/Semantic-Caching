"""Matrix core of the semantic cache, written from scratch.

Every linear-algebra step the project relies on lives in this one file, and the
rest of the package calls into it (``from semcache import linalg as la``):

    TF-IDF weighting      term_frequency_matrix, document_frequency,
                          inverse_document_frequency, tfidf_matrix
    Normalisation         row_norms, l2_normalize_rows, frobenius_norm_sq
    Orthogonalisation     gram_schmidt_qr
    Eigen-decomposition   jacobi_eigh                (symmetric matrices)
    SVD                   exact_svd                  (small, dense)
                          randomized_svd             (large, sparse)
                          truncated_svd              (picks one of the two)
    LSA projection        fold_in, project, zero_columns
    Latent coverage       latent_coverage            (how much of a row survives the SVD)
    Similarity            cosine_scores, cosine_matrix, rowwise_cosine, sparse_dot,
                          sparse_row_scores, blend_scores, guarded_scores, top_k
    Approximate search    RandomHyperplaneLSH, lsh_collision_probability

Ground rules
------------
* NumPy supplies arrays and the matrix product operator ``@``.
* ``scipy.sparse`` is used only to *store* the sparse TF-IDF matrix and to
  multiply it by dense blocks. Without it a 100k x 40k TF-IDF matrix would need
  ~32 GB as a dense array.
* ``numpy.linalg.svd / eig / eigh / qr`` are never called in this file.
  ``tests/test_linalg.py`` uses them as an independent check on our results.

Layout convention
-----------------
Rows are documents (queries), columns are terms, as in scikit-learn:

    X (n_docs x n_terms)  ~=  U_k  Sigma_k  V_k^T

so ``V_k`` (n_terms x k) maps a TF-IDF row into the latent space. Textbooks
often use the transpose, a terms x documents matrix A = X^T; then their U_k is
our V_k and the fold-in formula  q_hat = Sigma_k^{-1} U_k^T q  becomes, for a
row vector, q_hat = q V_k Sigma_k^{-1}.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "term_frequency_matrix",
    "document_frequency",
    "inverse_document_frequency",
    "tfidf_matrix",
    "row_norms",
    "l2_normalize_rows",
    "frobenius_norm_sq",
    "gram_schmidt_qr",
    "jacobi_eigh",
    "exact_svd",
    "randomized_svd",
    "truncated_svd",
    "fold_in",
    "project",
    "zero_columns",
    "latent_coverage",
    "cosine_scores",
    "cosine_matrix",
    "rowwise_cosine",
    "sparse_dot",
    "sparse_row_scores",
    "blend_scores",
    "guarded_scores",
    "top_k",
    "RandomHyperplaneLSH",
    "lsh_collision_probability",
]


# ---------------------------------------------------------------------------
# 1. TF-IDF
# ---------------------------------------------------------------------------

def term_frequency_matrix(docs: Sequence[Sequence[int]], n_terms: int) -> sp.csr_matrix:
    """Raw term counts. ``TF[d, t]`` = number of times term ``t`` occurs in document ``d``.

    ``docs`` is a list of documents, each a list of vocabulary indices.
    """
    n_docs = len(docs)
    lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=n_docs)
    total = int(lengths.sum())
    if total == 0:
        return sp.csr_matrix((n_docs, n_terms), dtype=np.float64)

    rows = np.repeat(np.arange(n_docs, dtype=np.int64), lengths)
    cols = np.fromiter(itertools.chain.from_iterable(docs), dtype=np.int64, count=total)
    if cols.min() < 0 or cols.max() >= n_terms:
        raise ValueError("term index out of range for the given n_terms")

    # Count repeated (row, col) pairs: encode each pair as one integer key.
    keys = rows * n_terms + cols
    unique_keys, counts = np.unique(keys, return_counts=True)
    r, c = np.divmod(unique_keys, n_terms)
    return sp.csr_matrix((counts.astype(np.float64), (r, c)), shape=(n_docs, n_terms))


def document_frequency(tf: sp.spmatrix) -> np.ndarray:
    """``df[t]`` = number of documents that contain term ``t`` at least once."""
    tf = sp.csr_matrix(tf)
    present = tf.indices[tf.data != 0]
    return np.bincount(present, minlength=tf.shape[1]).astype(np.int64)


def inverse_document_frequency(df: np.ndarray, n_docs: int, smooth: bool = True) -> np.ndarray:
    """Inverse document frequency.

    smooth=False:  idf(t) = ln(N / df(t))                (textbook form)
    smooth=True:   idf(t) = ln((1 + N) / (1 + df(t))) + 1 (scikit-learn's default)

    The smoothed form never divides by zero and keeps very common words
    slightly above zero instead of erasing them.
    """
    df = np.asarray(df, dtype=np.float64)
    if smooth:
        return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    idf = np.zeros_like(df)
    present = df > 0
    idf[present] = np.log(n_docs / df[present])
    return idf


def tfidf_matrix(
    tf: sp.spmatrix,
    idf: np.ndarray,
    sublinear_tf: bool = False,
    normalize: bool = True,
) -> sp.csr_matrix:
    """TF-IDF(d, t) = TF(d, t) * IDF(t), then each row scaled to unit length.

    ``sublinear_tf`` replaces TF by 1 + ln(TF), so a word repeated ten times
    does not count ten times as much.
    """
    X = sp.csr_matrix(tf, dtype=np.float64, copy=True)
    if sublinear_tf:
        X.data = 1.0 + np.log(X.data)
    X.data *= np.asarray(idf, dtype=np.float64)[X.indices]
    return l2_normalize_rows(X) if normalize else X


# ---------------------------------------------------------------------------
# 2. Norms and normalisation
# ---------------------------------------------------------------------------

def row_norms(X) -> np.ndarray:
    """Euclidean norm of every row, for dense or sparse matrices."""
    if sp.issparse(X):
        X = sp.csr_matrix(X)
        row_ids = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        squares = np.bincount(row_ids, weights=X.data * X.data, minlength=X.shape[0])
        return np.sqrt(squares)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.sqrt(np.einsum("ij,ij->i", X, X))


def l2_normalize_rows(X, eps: float = 1e-12):
    """Scale each row to unit length. All-zero rows stay zero.

    After this step the dot product of two rows *is* their cosine similarity.
    Accepts a 1-D vector too (returned as a 1-D vector).
    """
    if not sp.issparse(X) and np.ndim(X) == 1:
        return l2_normalize_rows(np.asarray(X, dtype=np.float64)[None, :], eps)[0]
    norms = row_norms(X)
    scale = np.zeros_like(norms)
    nonzero = norms > eps
    scale[nonzero] = 1.0 / norms[nonzero]
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float64, copy=True)
        X.data *= np.repeat(scale, np.diff(X.indptr))
        return X
    return np.asarray(X, dtype=np.float64) * scale[:, None]


def frobenius_norm_sq(X) -> float:
    """||X||_F^2 = sum of all squared entries = sum of all squared singular values."""
    if sp.issparse(X):
        data = sp.csr_matrix(X).data
        return float(data @ data)
    X = np.asarray(X, dtype=np.float64)
    return float(np.einsum("ij,ij->", X, X))


# ---------------------------------------------------------------------------
# 3. QR by Gram-Schmidt
# ---------------------------------------------------------------------------

def gram_schmidt_qr(A: np.ndarray, rank_tol: float = 1e-12):
    """Thin QR factorisation A = Q R with orthonormal columns in Q.

    Classical Gram-Schmidt applied twice per column ("twice is enough",
    Kahan-Parlett), which keeps Q orthogonal to machine precision while every
    projection stays a fast matrix-vector product.

    If a column is (numerically) a combination of the previous ones, its Q
    column is left as zeros and R[j, j] = 0; Q R = A still holds.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("gram_schmidt_qr expects a 2-D array")
    m, n = A.shape
    if n > m:
        raise ValueError(f"need at least as many rows as columns, got {A.shape}")

    Q = np.zeros((m, n), order="F")  # column-major: Q[:, j] is contiguous
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        original_norm = np.sqrt(v @ v)
        if j > 0:
            basis = Q[:, :j]
            for _ in range(2):
                coeffs = basis.T @ v      # projections onto earlier columns
                v -= basis @ coeffs       # remove them
                R[:j, j] += coeffs
        remaining = np.sqrt(v @ v)
        if original_norm == 0.0 or remaining <= rank_tol * original_norm:
            R[j, j] = 0.0                 # dependent column
            continue
        R[j, j] = remaining
        Q[:, j] = v / remaining
    return np.ascontiguousarray(Q), R


# ---------------------------------------------------------------------------
# 4. Symmetric eigen-decomposition by Jacobi rotations
# ---------------------------------------------------------------------------

def _round_robin_rounds(n: int):
    """Split all n(n-1)/2 index pairs into n-1 rounds of disjoint pairs.

    Same schedule as a round-robin tournament. Rotations on disjoint pairs do
    not interfere, so one whole round can be applied at once with NumPy.
    """
    players = list(range(n)) + ([-1] if n % 2 else [])  # -1 = "bye"
    size = len(players)
    rounds = []
    for _ in range(size - 1):
        p_list, q_list = [], []
        for i in range(size // 2):
            a, b = players[i], players[size - 1 - i]
            if a >= 0 and b >= 0:
                p_list.append(min(a, b))
                q_list.append(max(a, b))
        rounds.append((np.array(p_list, dtype=np.intp), np.array(q_list, dtype=np.intp)))
        players = [players[0], players[-1]] + players[1:-1]
    return rounds


def _off_diagonal_norm(A: np.ndarray) -> float:
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.sqrt(np.einsum("ij,ij->", off, off)))


def jacobi_eigh(S: np.ndarray, tol: float = 1e-12, max_sweeps: int = 60):
    """Eigenvalues and eigenvectors of a real symmetric matrix, S = W diag(w) W^T.

    Jacobi's method: repeatedly pick an off-diagonal entry S[p, q] and rotate
    rows/columns p and q by the angle that makes it exactly zero,

        S  <-  J^T S J,      J = rotation in the (p, q) plane,

    with tan(theta) = t, where  tau = (S[q,q] - S[p,p]) / (2 S[p,q])  and
    t = sign(tau) / (|tau| + sqrt(1 + tau^2)). Each rotation keeps the
    eigenvalues, and the off-diagonal mass shrinks until S is diagonal. The
    product of all rotations is the eigenvector matrix W.

    Returns ``(w, W)`` with eigenvalues sorted in descending order.
    """
    A = np.array(S, dtype=np.float64, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("jacobi_eigh expects a square matrix")
    scale = float(np.abs(A).max()) if A.size else 0.0
    if not np.allclose(A, A.T, atol=1e-9 * max(scale, 1.0)):
        raise ValueError("jacobi_eigh expects a symmetric matrix")
    n = A.shape[0]
    A = 0.5 * (A + A.T)
    W = np.eye(n)
    if n <= 1 or scale == 0.0:
        return A.diagonal().copy(), W

    frob = float(np.sqrt(np.einsum("ij,ij->", A, A)))
    rounds = _round_robin_rounds(n)
    converged = False
    for _ in range(max_sweeps):
        if _off_diagonal_norm(A) <= tol * frob:
            converged = True
            break
        for P, Q in rounds:
            if P.size == 0:
                continue
            apq = A[P, Q]
            active = np.abs(apq) > 1e-300
            if not active.any():
                continue
            P, Q, apq = P[active], Q[active], apq[active]
            tau = (A[Q, Q] - A[P, P]) / (2.0 * apq)
            t = np.sign(tau) / (np.abs(tau) + np.sqrt(1.0 + tau * tau))
            t[tau == 0.0] = 1.0           # 45-degree rotation
            c = 1.0 / np.sqrt(1.0 + t * t)
            s = t * c

            # A <- A J   (mix columns p and q)
            col_p, col_q = A[:, P], A[:, Q]           # fancy indexing copies
            A[:, P] = col_p * c - col_q * s
            A[:, Q] = col_p * s + col_q * c
            # A <- J^T A (mix rows p and q)
            row_p, row_q = A[P, :], A[Q, :]
            A[P, :] = c[:, None] * row_p - s[:, None] * row_q
            A[Q, :] = s[:, None] * row_p + c[:, None] * row_q
            A[P, Q] = 0.0                              # exactly zero by construction
            A[Q, P] = 0.0
            # W <- W J   (accumulate the eigenvectors)
            w_p, w_q = W[:, P], W[:, Q]
            W[:, P] = w_p * c - w_q * s
            W[:, Q] = w_p * s + w_q * c
    if not converged and _off_diagonal_norm(A) > tol * frob:
        warnings.warn("jacobi_eigh: did not fully converge; results may be inaccurate")

    w = A.diagonal().copy()
    order = np.argsort(w)[::-1]
    return w[order], W[:, order]


# ---------------------------------------------------------------------------
# 5. Singular value decomposition
# ---------------------------------------------------------------------------

def exact_svd(A: np.ndarray, k: int | None = None):
    """Thin SVD of a small dense matrix, A = U diag(s) V^T.

    Uses the classic identity: if A^T A = V diag(s^2) V^T (an eigen-
    decomposition), then the singular values are s and U = A V diag(1/s).
    The Gram matrix of the *shorter* side is used, so its size is
    min(m, n) x min(m, n).

    Note: squaring A also squares its condition number, so the smallest
    singular values come out less accurately than with a dedicated SVD
    routine. For LSA we only keep the largest ones, where this is harmless.

    Returns ``(U, s, V)``, singular values in descending order. Pass ``k`` to
    keep only the top-k triplets.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m >= n:
        w, V = jacobi_eigh(A.T @ A)
        s = np.sqrt(np.clip(w, 0.0, None))
        U = np.zeros((m, n))
        nz = s > (s[0] * 1e-12 if s.size and s[0] > 0 else np.inf)
        U[:, nz] = (A @ V[:, nz]) / s[nz]
    else:
        w, U = jacobi_eigh(A @ A.T)
        s = np.sqrt(np.clip(w, 0.0, None))
        V = np.zeros((n, m))
        nz = s > (s[0] * 1e-12 if s.size and s[0] > 0 else np.inf)
        V[:, nz] = (A.T @ U[:, nz]) / s[nz]
    if k is not None:
        U, s, V = U[:, :k], s[:k], V[:, :k]
    return U, s, V


def _randomized_svd_core(A, k: int, oversample: int, power_iters: int, seed: int):
    """Randomized SVD for an m x n matrix with m <= n (orthonormalise on the short side)."""
    m, n = A.shape
    width = min(k + oversample, m)
    rng = np.random.default_rng(seed)

    # Step 1: sample the range of A with a random test matrix.
    omega = rng.standard_normal((n, width))
    Q, _ = gram_schmidt_qr(np.asarray(A @ omega))

    # Step 2: power (subspace) iterations sharpen the gap between kept and
    # discarded singular values: the sample becomes (A A^T)^q A omega.
    for _ in range(power_iters):
        Q, _ = gram_schmidt_qr(np.asarray(A @ np.asarray(A.T @ Q)))

    # Step 3: project A onto the basis, B = Q^T A (width x n). We store B^T.
    Bt = np.asarray(A.T @ Q)

    # Step 4: SVD of the small B through the eigen-decomposition of B B^T.
    w, W = jacobi_eigh(Bt.T @ Bt)
    s = np.sqrt(np.clip(w, 0.0, None))
    U = Q @ W
    V = np.zeros((n, width))
    nz = s > (s[0] * 1e-12 if s.size and s[0] > 0 else np.inf)
    V[:, nz] = (Bt @ W[:, nz]) / s[nz]
    return U[:, :k], s[:k], V[:, :k]


def randomized_svd(A, k: int, oversample: int = 20, power_iters: int = 7, seed: int = 0):
    """Truncated SVD  A ~= U_k diag(s_k) V_k^T  for large (sparse) matrices.

    Halko, Martinsson & Tropp (2011), "Finding structure with randomness":
      1. Y = A Omega with a Gaussian Omega of k + oversample columns
      2. a few power iterations  Y <- A (A^T Y)
      3. Q = orthonormal basis of Y (our Gram-Schmidt)
      4. B = Q^T A is small; its SVD comes from jacobi_eigh(B B^T)
      5. U = Q U_B, and V from B

    Works for dense arrays and scipy.sparse matrices. All the heavy work is
    matrix products, so it never forms a dense copy of a sparse A.

    Returns ``(U, s, V)``: U is m x k, s has length k, V is n x k.
    """
    m, n = A.shape
    if not 1 <= k <= min(m, n):
        raise ValueError(f"k must be between 1 and min(m, n) = {min(m, n)}, got {k}")
    if m > n:
        # Run on A^T so Gram-Schmidt works on the short side, then swap U and V.
        V, s, U = _randomized_svd_core(A.T, k, oversample, power_iters, seed)
        return U, s, V
    return _randomized_svd_core(A, k, oversample, power_iters, seed)


def truncated_svd(A, k: int, oversample: int = 20, power_iters: int = 7, seed: int = 0,
                  exact_limit: int = 300):
    """Top-k SVD. If the shorter side is at most ``exact_limit`` the matrix is small
    enough to densify, so ``exact_svd`` is used; otherwise ``randomized_svd``."""
    if min(A.shape) <= exact_limit:
        dense = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=np.float64)
        return exact_svd(dense, k)
    return randomized_svd(A, k, oversample=oversample, power_iters=power_iters, seed=seed)


# ---------------------------------------------------------------------------
# 6. Latent semantic analysis: placing new queries in the latent space
# ---------------------------------------------------------------------------

def fold_in(X_rows, V_k: np.ndarray, s_k: np.ndarray) -> np.ndarray:
    """Latent coordinates of new TF-IDF rows:  q_hat = q V_k Sigma_k^{-1}.

    For a row of the training matrix this returns exactly its row of U_k,
    because X V_k = U_k Sigma_k. That is how a query the SVD has never seen
    is placed in the same space as the training queries, without recomputing
    the SVD. Directions with a zero singular value get coordinate 0.
    """
    s_k = np.asarray(s_k, dtype=np.float64)
    inv = np.zeros_like(s_k)
    nonzero = s_k > (s_k.max() * 1e-12 if s_k.size else 0.0)
    inv[nonzero] = 1.0 / s_k[nonzero]
    return np.asarray(X_rows @ V_k) * inv


def project(X_rows, V_k: np.ndarray, s_k: np.ndarray, sigma_power: float = 1.0) -> np.ndarray:
    """Embedding used for similarity: q_hat Sigma_k^p.

    p = 1 gives q V_k, the plain projection onto the top-k directions
          (what scikit-learn's TruncatedSVD.transform returns);
    p = 0 gives the fold-in coordinates, where every latent direction
          counts equally.
    """
    return fold_in(X_rows, V_k, s_k) * (s_k ** sigma_power)


def zero_columns(X, keep: np.ndarray):
    """X with the columns where ``keep`` is False set to zero: X diag(keep).

    Used to leave chosen terms (e.g. "what", "how") out of the SVD fit while they
    stay in the TF-IDF rows used for exact and word-overlap scoring.
    """
    keep = np.asarray(keep, dtype=np.float64)
    if sp.issparse(X):
        return sp.csr_matrix(X @ sp.diags(keep))
    return np.asarray(X, dtype=np.float64) * keep


def latent_coverage(X_rows, V_k: np.ndarray, keep: np.ndarray | None = None,
                    eps: float = 1e-12) -> np.ndarray:
    """rho(x) = ||x V_k|| / ||x_svd||: how much of each row survives in the k-dim space.

    V_k has orthonormal columns, so x V_k is the projection of x onto the span of the
    top-k directions and 0 <= rho <= 1. rho near 1: the question lives in the space
    the SVD kept. rho near 0: almost all of it was thrown away, and its latent vector
    is decided by the few words that survived (often just "what" or "how").

    ``keep`` marks the terms the SVD was fitted on; terms left out of the fit (their
    rows of V_k are zero) are left out of the denominator too, so they don't count
    as "lost".
    """
    Z = np.asarray(X_rows @ V_k)
    num = np.sqrt(np.sum(Z * Z, axis=1))
    den = row_norms(zero_columns(X_rows, keep) if keep is not None else X_rows)
    out = np.zeros_like(num)
    ok = den > eps
    out[ok] = num[ok] / den[ok]
    return np.minimum(out, 1.0)


# ---------------------------------------------------------------------------
# 7. Cosine similarity
# ---------------------------------------------------------------------------

def cosine_scores(C: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Similarity of one query to every cached vector in a single product.

    With the rows of C and the vector q already unit length,
        cos(c_i, q) = c_i . q,   so all scores = C q.
    """
    return np.asarray(C @ q).ravel()


def cosine_matrix(Q_rows: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Similarities of many queries against the cache at once: S = Q C^T."""
    return np.asarray(Q_rows @ C.T)


def rowwise_cosine(A, B) -> np.ndarray:
    """cos(a_i, b_i) for matching rows of A and B (dense or sparse)."""
    A_n, B_n = l2_normalize_rows(A), l2_normalize_rows(B)
    if sp.issparse(A_n) or sp.issparse(B_n):
        return np.asarray(sp.csr_matrix(A_n).multiply(B_n).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", A_n, B_n)


def sparse_dot(idx_a: np.ndarray, val_a: np.ndarray, idx_b: np.ndarray, val_b: np.ndarray) -> float:
    """Dot product of two sparse vectors stored as (sorted indices, values).

    Only coordinates present in both vectors contribute, so this costs
    O(nnz log nnz) instead of O(vocabulary size). For unit-length TF-IDF rows
    the result is their cosine similarity.
    """
    if idx_a.size == 0 or idx_b.size == 0:
        return 0.0
    pos = np.searchsorted(idx_b, idx_a)
    pos_clipped = np.minimum(pos, idx_b.size - 1)
    common = idx_b[pos_clipped] == idx_a
    return float(val_a[common] @ val_b[pos_clipped[common]])


def sparse_row_scores(C_rows, q_row) -> np.ndarray:
    """Word-overlap cosine of one sparse unit row q against every sparse unit row of C:
    C q^T, the TF-IDF analogue of cosine_scores."""
    C_rows = sp.csr_matrix(C_rows)
    q_row = sp.csr_matrix(q_row)
    if C_rows.shape[0] == 0:
        return np.zeros(0)
    width = max(C_rows.shape[1], q_row.shape[1])      # columns past the last stored term are zero
    if C_rows.shape[1] != width:
        C_rows = sp.csr_matrix((C_rows.data, C_rows.indices, C_rows.indptr), shape=(C_rows.shape[0], width))
    if q_row.shape[1] != width:
        q_row = sp.csr_matrix((q_row.data, q_row.indices, q_row.indptr), shape=(q_row.shape[0], width))
    return np.asarray((C_rows @ q_row.T).todense()).ravel()


def blend_scores(lsa: np.ndarray, lexical: np.ndarray, lsa_weight: float) -> np.ndarray:
    """Hybrid similarity  w * cos_LSA + (1 - w) * cos_TFIDF.

    LSA captures synonyms ("cheap" ~ "low cost"); raw TF-IDF keeps rare, exact
    words (names, versions) that the truncated SVD may have discarded.
    """
    return lsa_weight * np.asarray(lsa) + (1.0 - lsa_weight) * np.asarray(lexical)


def guarded_scores(lsa: np.ndarray, lexical: np.ndarray, pair_rho: np.ndarray, lsa_weight: float,
                   min_coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """The score the hit decision uses, with the latent-coverage guard.

    A pair is *blocked* when the smaller rho of its two questions is below
    ``min_coverage``: too little of one of them survives the SVD for its latent
    cosine to mean anything. A blocked pair is scored by word overlap alone (the
    TF-IDF cosine); every other pair gets the usual blend. Returns (scores, blocked).
    """
    blended = blend_scores(lsa, lexical, lsa_weight)
    blocked = np.asarray(pair_rho) < min_coverage
    return np.where(blocked, np.asarray(lexical, dtype=np.float64), blended), blocked


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first (partial sort, O(n))."""
    scores = np.asarray(scores).ravel()
    k = min(k, scores.size)
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


# ---------------------------------------------------------------------------
# 8. Locality-sensitive hashing with random hyperplanes
# ---------------------------------------------------------------------------

class RandomHyperplaneLSH:
    """Approximate nearest-neighbour index for cosine similarity.

    Each of ``n_tables`` tables draws ``n_bits`` random hyperplane normals
    r_1 ... r_b and hashes a vector x to its sign pattern

        h(x) = [ r_1 . x > 0, ..., r_b . x > 0 ]   (packed into one integer).

    Two vectors at angle theta land on the same side of a random hyperplane
    with probability 1 - theta / pi, so similar vectors tend to share a bucket.
    A lookup only compares the query against vectors in its buckets, instead
    of the whole cache. More bits per table = fewer, purer candidates; more
    tables = fewer missed neighbours.
    """

    def __init__(self, dim: int, n_bits: int = 12, n_tables: int = 8, seed: int = 0):
        if not 1 <= n_bits <= 62:
            raise ValueError("n_bits must be between 1 and 62")
        rng = np.random.default_rng(seed)
        self.dim, self.n_bits, self.n_tables = dim, n_bits, n_tables
        self.planes = rng.standard_normal((n_tables, dim, n_bits))
        self._bit_weights = np.left_shift(1, np.arange(n_bits, dtype=np.int64))
        self.tables: list[dict[int, set[int]]] = [{} for _ in range(n_tables)]
        self._keys: dict[int, np.ndarray] = {}

    def hash(self, X: np.ndarray) -> np.ndarray:
        """Bucket keys, shape (n_vectors, n_tables)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        bits = np.einsum("nd,tdb->ntb", X, self.planes) > 0
        return bits.astype(np.int64) @ self._bit_weights

    def add(self, ids: Iterable[int], X: np.ndarray) -> None:
        ids = list(ids)
        keys = self.hash(X)
        for item_id, row in zip(ids, keys):
            if item_id in self._keys:
                self.remove(item_id)
            for table, key in zip(self.tables, row):
                table.setdefault(int(key), set()).add(item_id)
            self._keys[item_id] = row

    def remove(self, item_id: int) -> None:
        row = self._keys.pop(item_id, None)
        if row is None:
            return
        for table, key in zip(self.tables, row):
            bucket = table.get(int(key))
            if bucket is not None:
                bucket.discard(item_id)
                if not bucket:
                    del table[int(key)]

    def candidates(self, q: np.ndarray) -> np.ndarray:
        """Ids sharing at least one bucket with q."""
        found: set[int] = set()
        for table, key in zip(self.tables, self.hash(q)[0]):
            found.update(table.get(int(key), ()))
        return np.fromiter(found, dtype=np.int64, count=len(found))

    def clear(self) -> None:
        self.tables = [{} for _ in range(self.n_tables)]
        self._keys.clear()

    def __len__(self) -> int:
        return len(self._keys)


def lsh_collision_probability(cos_sim, n_bits: int, n_tables: int):
    """Chance that a vector with cosine similarity ``cos_sim`` to the query is a candidate.

        p_bit   = 1 - arccos(cos_sim) / pi
        p_table = p_bit ** n_bits
        p_found = 1 - (1 - p_table) ** n_tables
    """
    theta = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    p_bit = 1.0 - theta / np.pi
    return 1.0 - (1.0 - p_bit ** n_bits) ** n_tables
