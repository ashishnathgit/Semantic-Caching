"""Checks every routine in semcache/linalg.py against an independent reference.

NumPy's LAPACK-backed numpy.linalg routines are used only here, as the answer key.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from semcache import linalg as la

RNG = np.random.default_rng(1234)


# --- TF-IDF ------------------------------------------------------------------

TOY_DOCS = [  # cheap flight goa / cheap train mumbai / flight mumbai   ("to" kept)
    ["cheap", "flights", "to", "goa"],
    ["cheap", "train", "to", "mumbai"],
    ["flights", "to", "mumbai"],
]
TOY_VOCAB = ["cheap", "flights", "to", "goa", "train", "mumbai"]


def toy_tf():
    index = {t: i for i, t in enumerate(TOY_VOCAB)}
    return la.term_frequency_matrix([[index[w] for w in d] for d in TOY_DOCS], len(TOY_VOCAB))


def test_term_frequency_counts_repeats():
    tf = la.term_frequency_matrix([[0, 0, 2], [], [1]], 3).toarray()
    assert tf.tolist() == [[2, 0, 1], [0, 0, 0], [0, 1, 0]]


def test_document_frequency_and_textbook_idf_match_hand_calculation():
    df = la.document_frequency(toy_tf())
    assert df.tolist() == [2, 2, 3, 1, 1, 2]
    idf = la.inverse_document_frequency(df, n_docs=3, smooth=False)
    expected = np.log(3 / np.array([2, 2, 3, 1, 1, 2]))
    np.testing.assert_allclose(idf, expected)
    assert idf[TOY_VOCAB.index("to")] == 0.0          # in every document -> erased


def test_smoothed_idf_formula():
    df = np.array([1, 2, 3])
    np.testing.assert_allclose(la.inverse_document_frequency(df, 3, smooth=True),
                               np.log(4 / (1 + df)) + 1)


def test_tfidf_rows_are_unit_length_and_match_hand_values():
    tf = toy_tf()
    idf = la.inverse_document_frequency(la.document_frequency(tf), 3, smooth=False)
    X_raw = la.tfidf_matrix(tf, idf, normalize=False).toarray()
    # d1 = (cheap 0.405, flights 0.405, goa 1.099), "to" contributes nothing
    np.testing.assert_allclose(X_raw[0], [np.log(1.5), np.log(1.5), 0, np.log(3), 0, 0])
    X = la.tfidf_matrix(tf, idf).toarray()
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0)


def test_tfidf_matches_scikit_learn_when_available():
    sklearn = pytest.importorskip("sklearn.feature_extraction.text")
    texts = [" ".join(d) for d in TOY_DOCS]
    vec = sklearn.TfidfVectorizer(vocabulary=TOY_VOCAB, token_pattern=r"\S+", sublinear_tf=True)
    expected = vec.fit_transform(texts).toarray()
    tf = toy_tf()
    idf = la.inverse_document_frequency(la.document_frequency(tf), 3, smooth=True)
    ours = la.tfidf_matrix(tf, idf, sublinear_tf=True).toarray()
    np.testing.assert_allclose(ours, expected, atol=1e-12)


# --- norms -------------------------------------------------------------------

def test_l2_normalize_dense_sparse_and_zero_rows():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    dense = la.l2_normalize_rows(X)
    sparse = la.l2_normalize_rows(sp.csr_matrix(X)).toarray()
    expected = np.array([[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(dense, expected)
    np.testing.assert_allclose(sparse, expected)
    np.testing.assert_allclose(la.l2_normalize_rows(np.array([0.0, 2.0])), [0.0, 1.0])


def test_frobenius_norm_equals_sum_of_squared_singular_values():
    A = RNG.standard_normal((30, 12))
    s = np.linalg.svd(A, compute_uv=False)
    assert la.frobenius_norm_sq(A) == pytest.approx(np.sum(s ** 2))
    assert la.frobenius_norm_sq(sp.csr_matrix(A)) == pytest.approx(np.sum(s ** 2))


# --- QR ----------------------------------------------------------------------

def test_gram_schmidt_qr_is_orthonormal_and_reconstructs():
    A = RNG.standard_normal((200, 25))
    Q, R = la.gram_schmidt_qr(A)
    np.testing.assert_allclose(Q.T @ Q, np.eye(25), atol=1e-12)
    np.testing.assert_allclose(Q @ R, A, atol=1e-12)
    assert np.allclose(R, np.triu(R))


def test_gram_schmidt_handles_dependent_columns():
    A = RNG.standard_normal((50, 3))
    A = np.column_stack([A, A[:, 0] + 2 * A[:, 1]])     # 4th column is dependent
    Q, R = la.gram_schmidt_qr(A)
    assert R[3, 3] == 0.0
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)


# --- Jacobi eigen-decomposition --------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 40, 61])
def test_jacobi_matches_numpy_eigh(n):
    M = RNG.standard_normal((n, n))
    S = M + M.T
    w, W = la.jacobi_eigh(S)
    np.testing.assert_allclose(w, np.linalg.eigvalsh(S)[::-1], atol=1e-10)
    np.testing.assert_allclose(W.T @ W, np.eye(n), atol=1e-10)
    np.testing.assert_allclose(S @ W, W * w, atol=1e-9)


def test_jacobi_rejects_non_symmetric():
    with pytest.raises(ValueError):
        la.jacobi_eigh(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_jacobi_handles_repeated_eigenvalues():
    Q, _ = np.linalg.qr(RNG.standard_normal((6, 6)))
    S = Q @ np.diag([5, 5, 5, 1, 1, 0]) @ Q.T
    w, W = la.jacobi_eigh(S)
    np.testing.assert_allclose(w, [5, 5, 5, 1, 1, 0], atol=1e-10)
    np.testing.assert_allclose(W @ np.diag(w) @ W.T, S, atol=1e-10)


# --- SVD ---------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(40, 15), (15, 40), (25, 25)])
def test_exact_svd_matches_numpy(shape):
    A = RNG.standard_normal(shape)
    U, s, V = la.exact_svd(A)
    np.testing.assert_allclose(s, np.linalg.svd(A, compute_uv=False), atol=1e-10)
    np.testing.assert_allclose(U @ np.diag(s) @ V.T, A, atol=1e-9)


def low_rank_plus_noise(m, n, rank, noise=1e-3):
    U, _ = np.linalg.qr(RNG.standard_normal((m, rank)))
    V, _ = np.linalg.qr(RNG.standard_normal((n, rank)))
    sigma = 10.0 * 0.7 ** np.arange(rank)
    return (U * sigma) @ V.T + noise * RNG.standard_normal((m, n))


@pytest.mark.parametrize("shape", [(500, 120), (120, 500)])
def test_randomized_svd_recovers_top_singular_values(shape):
    A = low_rank_plus_noise(*shape, rank=12)
    U, s, V = la.randomized_svd(A, k=8, seed=0)
    np.testing.assert_allclose(s, np.linalg.svd(A, compute_uv=False)[:8], rtol=1e-8)
    np.testing.assert_allclose(U.T @ U, np.eye(8), atol=1e-10)
    np.testing.assert_allclose(V.T @ V, np.eye(8), atol=1e-10)


def test_randomized_svd_is_near_optimal_eckart_young():
    """The rank-k reconstruction error should be within 0.1% of the best possible one."""
    A = low_rank_plus_noise(300, 80, rank=20, noise=0.05)
    k = 10
    U, s, V = la.randomized_svd(A, k=k, seed=3)
    ours = np.linalg.norm(A - (U * s) @ V.T)
    best = np.sqrt(np.sum(np.linalg.svd(A, compute_uv=False)[k:] ** 2))
    assert ours <= best * 1.001


def test_randomized_svd_on_sparse_input():
    dense = low_rank_plus_noise(400, 90, rank=6, noise=0.0)
    dense[np.abs(dense) < 0.05] = 0.0
    A = sp.csr_matrix(dense)
    U, s, V = la.randomized_svd(A, k=5, seed=0)
    np.testing.assert_allclose(s, np.linalg.svd(dense, compute_uv=False)[:5], rtol=1e-8)


def test_randomized_svd_validates_k():
    with pytest.raises(ValueError):
        la.randomized_svd(RNG.standard_normal((5, 4)), k=5)


# --- LSA fold-in ---------------------------------------------------------------

def test_fold_in_of_a_training_row_gives_its_row_of_U():
    X = RNG.standard_normal((60, 30))
    U, s, V = la.exact_svd(X, k=6)
    np.testing.assert_allclose(la.fold_in(X[:5], V, s), U[:5], atol=1e-9)


def test_project_with_sigma_power_one_is_plain_projection():
    X = RNG.standard_normal((20, 10))
    U, s, V = la.exact_svd(X, k=4)
    np.testing.assert_allclose(la.project(X, V, s, sigma_power=1.0), X @ V, atol=1e-10)
    np.testing.assert_allclose(la.project(X, V, s, sigma_power=0.0), U, atol=1e-9)


# --- cosine and top-k ------------------------------------------------------------

def test_cosine_scores_match_definition():
    C = la.l2_normalize_rows(RNG.standard_normal((10, 5)))
    q = la.l2_normalize_rows(RNG.standard_normal(5))
    expected = [c @ q / (np.linalg.norm(c) * np.linalg.norm(q)) for c in C]
    np.testing.assert_allclose(la.cosine_scores(C, q), expected)
    np.testing.assert_allclose(la.cosine_matrix(q[None, :], C)[0], expected)


def test_rowwise_cosine_dense_and_sparse():
    A, B = RNG.standard_normal((6, 4)), RNG.standard_normal((6, 4))
    expected = np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    np.testing.assert_allclose(la.rowwise_cosine(A, B), expected)
    np.testing.assert_allclose(la.rowwise_cosine(sp.csr_matrix(A), sp.csr_matrix(B)), expected)


def test_top_k_orders_best_first():
    scores = np.array([0.1, 0.9, 0.3, 0.9, 0.5])
    assert la.top_k(scores, 3).tolist() == [1, 3, 4]
    assert la.top_k(scores, 10).size == 5
    assert la.top_k(np.array([]), 3).size == 0


# --- LSH ---------------------------------------------------------------------

def test_lsh_bit_agreement_matches_angle_formula():
    lsh = la.RandomHyperplaneLSH(dim=50, n_bits=60, n_tables=40, seed=0)
    x = la.l2_normalize_rows(RNG.standard_normal(50))
    r = RNG.standard_normal(50)
    noise = la.l2_normalize_rows(r - (r @ x) * x)  # unit vector orthogonal to x
    y = 0.8 * x + 0.6 * noise                    # cos(x, y) = 0.8 exactly
    bits_x = np.einsum("d,tdb->tb", x, lsh.planes) > 0
    bits_y = np.einsum("d,tdb->tb", y, lsh.planes) > 0
    agreement = np.mean(bits_x == bits_y)
    assert agreement == pytest.approx(1 - np.arccos(0.8) / np.pi, abs=0.04)


def test_lsh_finds_near_duplicates_and_supports_removal():
    data = la.l2_normalize_rows(RNG.standard_normal((500, 32)))
    lsh = la.RandomHyperplaneLSH(dim=32, n_bits=8, n_tables=10, seed=1)
    lsh.add(range(500), data)
    query = la.l2_normalize_rows(data[42] + 0.05 * RNG.standard_normal(32))
    assert 42 in lsh.candidates(query)
    assert len(lsh.candidates(query)) < 500            # fewer comparisons than brute force
    lsh.remove(42)
    assert 42 not in lsh.candidates(query)
    assert len(lsh) == 499


def test_collision_probability_increases_with_similarity():
    p = la.lsh_collision_probability(np.array([0.2, 0.5, 0.8, 0.95, 1.0]), n_bits=12, n_tables=8)
    assert np.all(np.diff(p) > 0)
    assert p[-1] == pytest.approx(1.0)


def test_sparse_dot_matches_dense_dot():
    a = sp.random(1, 200, density=0.05, random_state=1, format="csr")
    b = sp.random(1, 200, density=0.05, random_state=2, format="csr")
    b = b + a * 0.5                                   # guarantee some overlap
    a.sort_indices(); b = sp.csr_matrix(b); b.sort_indices()
    expected = float((a.toarray() @ b.toarray().T)[0, 0])
    assert la.sparse_dot(a.indices, a.data, b.indices, b.data) == pytest.approx(expected)
    assert la.sparse_dot(np.empty(0, int), np.empty(0), b.indices, b.data) == 0.0


def test_blend_scores_is_a_convex_combination():
    np.testing.assert_allclose(la.blend_scores([1.0, 0.0], [0.0, 1.0], 0.75), [0.75, 0.25])
