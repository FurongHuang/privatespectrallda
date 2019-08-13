from itertools import permutations
from functools import wraps

import numpy as np
import scipy.stats

SEED = 42


def noiser(edge=None, symmetric=False):
    def interm(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):

            result = f(self, *args, **kwargs).copy()

            total_noise = np.zeros(result.shape)
            if edge and edge in self.variances:
                if symmetric and len(result.shape) == 3:
                    dim = result.shape[0]
                    perms = list(permutations(range(3)))
                    noise = np.random.normal(0,
                                             self.variances[edge] / len(perms),
                                             size=(dim, dim, dim))

                    for perm in perms:
                        total_noise += np.transpose(noise, perm)

                else:
                    total_noise += np.random.normal(0, self.variances[edge],
                                                    size=result.shape)

                result += total_noise

            return result

        return wrapper

    return interm


def seed(seed_val=SEED):
    def interm(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            np.random.seed(seed_val)
            return f(*args, **kwargs)
        return wrapper
    return interm


def proj_l1_simplex(vec, l1_simplex_boundary=1):
    vec_sorted = -np.sort(-vec)
    vec_shifted = (vec_sorted - (vec_sorted.cumsum() - l1_simplex_boundary) /
                   range(1, len(vec) + 1))
    rho = np.squeeze(np.where(vec_shifted > 0)).max() + 1
    theta = (vec_sorted[:rho].sum() - l1_simplex_boundary) / rho
    return np.maximum(vec - theta, 0)


def flatten2(items):
    flattened = []
    for item in items:
        for subitem in item:
            flattened.append(subitem)
    return flattened

@seed(42)
def generate_document_word_counts(alpha, beta, n_docs, n_words_in_doc=(500, 10000), full_documents=False):
    """
    :param alpha: k*1 input
    :param beta: n_words*k input
    :param n_docs: Number of documents.
    :param n_words_in_doc: Range of number of words in a given doc
    :return: n_docs*n_words document_word_counts matrix
    """

    k = len(alpha)
    n_words, also_k = beta.shape
    n_words_in_doc_min, n_words_in_doc_max = n_words_in_doc

    # Draw n_docs samples from a Dirichlet characterized by alpha -
    # n_docs*k matrix.
    d_t_weights = scipy.stats.dirichlet.rvs(alpha, size=n_docs, random_state=42)

    # Draw word counts from a range for each document - n_docs*1 matrix
    d_n_words = np.random.randint(n_words_in_doc_min,
                                  n_words_in_doc_max,
                                  size=n_docs)

    d_w_counts = np.array([scipy.stats.multinomial.rvs(d_n_word, beta.dot(d_t_weight)) for d_n_word, d_t_weight in zip(d_n_words, d_t_weights)])

    if full_documents:
        return [' '.join(flatten2([[str(i)] * int(word) for i, word in enumerate(doc)])) for doc in d_w_counts]

    return d_w_counts