from typing import Dict

import numpy as np
import scipy.linalg
import tensorly
from tensorly.decomposition import parafac

from .helpers import noiser, proj_l1_simplex


class DPSLatentDirichletAllocation:
    def __init__(
            self,
            n_topics: int,
            doc_topic_prior,
            variances: Dict = dict(),
            l1_simplex_projection: bool = True,
    ):
        """

        :param n_topics: Number of topics.
        :param doc_topic_prior: Prior of document topic distribution. Also called alpha.
        :param variances: Noise variances configured to a given set of edges.
        :param l1_simplex_projection: Boolean condition illustrating projection of beta to simplex.
        """
        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.variances = variances
        self.l1_simplex_projection = l1_simplex_projection

    def __repr__(self):
        return f'{self.__class__.__name__}(n_topics={self.n_topics}, ' \
            f'doc_topic_prior={self.doc_topic_prior})'

    def fit(self, document_word_counts):
        """

        :param document_word_counts: Document Word Count Matrix
        :return:
        """
        self.document_word_counts = document_word_counts

        self.m2_eigenvalues, self.m2_eigenvectors = self.decompose_moment2()
        self.whitener = self.create_whitener()
        self.unwhitener = self.create_unwhitener()

        if 'e3' in self.variances:
            self.moment3 = self.create_moment3()
            self.whitened_moment3 = self.whiten_moment3()
        else:
            self.whitened_moment3 = self.create_whitened_moment3()

        self.factors = self.decompose_moment3()
        self.m3_eigenvalues = np.linalg.norm(self.factors[0], axis=0)
        self.factors[0] /= self.m3_eigenvalues

        self.unique_factor = self.factor_correct_sign()

        self.doc_topic_posterior = (1 / (self.m3_eigenvalues ** 2))[::-1]
        self.topic_word_distribution = self.create_topic_word_distribution()

        if self.l1_simplex_projection:
            for i in range(self.n_topics):
                self.topic_word_distribution[:, i] = proj_l1_simplex(self.topic_word_distribution[:, i])

        self.topic_word_distribution = self.topic_word_distribution[:, ::-1]

    @property
    def moment1(self):
        """
        First moment of document word count matrix.
        :return:
        """
        return self.document_word_proportions.sum(axis=0) / self.n_docs

    # @seed(44)
    def decompose_moment2(self):
        """
        This method approximates the second moment by using power iteration
        and computes a truncated SVD on it.
        :return:
        """

        def approximate_moment2(x):
            p1t1 = np.einsum('ij,ik->jk', scaled_document_word_counts,
                             self.document_word_counts.dot(x))
            p1t2 = np.diag(scaled_document_word_counts.sum(axis=0)).dot(x)
            p1 = (p1t1 - p1t2) / self.n_docs

            p2 = (self.alpha0 / (self.alpha0 + 1)) * np.outer(self.moment1,
                                                              self.moment1.dot(x))

            return p1 - p2

        n_topics_augmented = np.min((self.n_topics + 5, self.n_words))

        x = np.random.randn(self.n_words, n_topics_augmented)

        scaling_factor = 1 / (self.l_ns * (self.l_ns - 1))
        scaled_document_word_counts = scaling_factor * self.document_word_counts

        for i in range(8):
            m2_approx = approximate_moment2(x)
            x, _ = scipy.linalg.qr(m2_approx, mode='economic')

        m2_approx = self.alpha0 * (self.alpha0 + 1) * approximate_moment2(x)
        U, s, Vh = scipy.linalg.svd(np.einsum('ij,ik->jk', m2_approx, m2_approx))

        m2_eigenvalues = np.sqrt(s)
        m2_eigenvectors = (x.dot(U)).T

        return m2_eigenvalues, m2_eigenvectors

    def create_whitened_moment3(self):
        def term1():
            scaling_factor = 1 / (self.l_ns * (self.l_ns - 1) * (self.l_ns - 2))
            scaled_document_word_counts = scaling_factor * self.whitened_document

            part1 = np.einsum('ij,ik,il->jkl', scaled_document_word_counts,
                              self.whitened_document,
                              self.whitened_document)

            rho = np.einsum('ij,ik->jk', scaled_document_word_counts,
                            self.whitened_document)
            part2 = np.zeros((self.n_words, self.n_words, self.n_words))
            diagonal_indices = np.diag_indices(self.n_words, ndim=2)
            for i, item in enumerate(rho):
                part2[i][diagonal_indices] = item
            part2 += np.einsum('ijk->jik', part2) + np.einsum('ijk->kji', part2)

            part3 = np.zeros((self.n_words, self.n_words, self.n_words))
            part3[np.diag_indices(self.n_words,
                                  ndim=3)] = 2 * scaled_document_word_counts.sum(
                axis=0)

            return (part1 - part2 + part3) / self.n_docs

        def term2():
            scaling_factor = 1 / (self.l_ns * (self.l_ns - 1))
            scaled_document_word_counts = scaling_factor * self.whitened_document

            part1 = np.einsum('i,jk,jl->ikl', 
                              self.whitened_moment1,
                              self.whitened_document,
                              self.whitened_document)
            part1 += np.einsum('ijk->kij', part1) + np.einsum('ijk->jki', part1)

            rho = np.einsum('i,j->ij', scaled_document_word_counts.sum(axis=0),
                            self.whitened_moment1)
            part2 = np.zeros((self.n_words, self.n_words, self.n_words))
            diagonal_indices = np.diag_indices(self.n_words, ndim=2)
            for i, item in enumerate(rho):
                part2[i][diagonal_indices] = item
            part2 += np.einsum('ijk->jik', part2) + np.einsum('ijk->kji', part2)

            return (self.alpha0 / (self.alpha0 + 2)) * (
                    part1 - part2) / self.n_docs

        def term3():
            return 2 * ((self.alpha0 ** 2) / (
                    (self.alpha0 + 1) * (self.alpha0 + 2))) * np.einsum(
                'i,j,k->ijk', 
                self.whitened_moment1,
                self.whitened_moment1,
                self.whitened_moment1)

        return (self.alpha0 * (self.alpha0 + 1) * (self.alpha0 + 2) / 2) * (
                term1() - term2() + term3())

    # @seed(43)
    @noiser(edge='e3')
    def create_moment3(self):
        """
        Creates the third moment.
        :return:
        """
        def term1():
            scaling_factor = 1 / (self.l_ns * (self.l_ns - 1) * (self.l_ns - 2))
            scaled_document_word_counts = scaling_factor * self.document_word_counts

            part1 = np.einsum('ij,ik,il->jkl', scaled_document_word_counts,
                              self.document_word_counts,
                              self.document_word_counts)

            rho = np.einsum('ij,ik->jk', scaled_document_word_counts,
                            self.document_word_counts)
            part2 = np.zeros((self.n_words, self.n_words, self.n_words))
            diagonal_indices = np.diag_indices(self.n_words, ndim=2)
            for i, item in enumerate(rho):
                part2[i][diagonal_indices] = item
            part2 += np.einsum('ijk->jik', part2) + np.einsum('ijk->kji', part2)

            part3 = np.zeros((self.n_words, self.n_words, self.n_words))
            part3[np.diag_indices(self.n_words,
                                  ndim=3)] = 2 * scaled_document_word_counts.sum(
                axis=0)

            return (part1 - part2 + part3) / self.n_docs

        def term2():
            scaling_factor = 1 / (self.l_ns * (self.l_ns - 1))
            scaled_document_word_counts = scaling_factor * self.document_word_counts

            part1 = np.einsum('i,jk,jl->ikl', self.moment1,
                              scaled_document_word_counts,
                              self.document_word_counts)
            part1 += np.einsum('ijk->kij', part1) + np.einsum('ijk->jki', part1)

            rho = np.einsum('i,j->ij', scaled_document_word_counts.sum(axis=0),
                            self.moment1)
            part2 = np.zeros((self.n_words, self.n_words, self.n_words))
            diagonal_indices = np.diag_indices(self.n_words, ndim=2)
            for i, item in enumerate(rho):
                part2[i][diagonal_indices] = item
            part2 += np.einsum('ijk->jik', part2) + np.einsum('ijk->kji', part2)

            return (self.alpha0 / (self.alpha0 + 2)) * (
                    part1 - part2) / self.n_docs

        def term3():
            moment1 = self.moment1
            return 2 * ((self.alpha0 ** 2) / (
                    (self.alpha0 + 1) * (self.alpha0 + 2))) * np.einsum(
                'i,j,k->ijk', moment1,
                moment1, moment1)

        return (self.alpha0 * (self.alpha0 + 1) * (self.alpha0 + 2) / 2) * (
                term1() - term2() + term3())

    # @seed(48)
    @noiser(edge='e6', symmetric=True)
    def whiten_moment3(self):
        """
        Whitens the third moment.
        :return:
        """
        return tensorly.tenalg.multi_mode_dot(self.moment3, np.array(
            [self.whitener.T for _ in range(3)]))

    # @seed(49)
    @noiser(edge='e7')
    def decompose_moment3(self):
        """
        Performs CP decomposition on third moment.
        :return:
        """
        return np.sort(np.array(parafac(self.whitened_moment3, self.n_topics)))[::-1]

    def factor_correct_sign(self):
        """
        Magic
        """
        factor = np.zeros((self.n_topics, self.n_topics))
        for i in range(self.n_topics):
            diff = [
                np.linalg.norm(self.factors[1][:, i] - self.factors[2][:, i]),
                np.linalg.norm(self.factors[0][:, i] - self.factors[2][:, i]),
                np.linalg.norm(self.factors[0][:, i] - self.factors[1][:, i]),
            ]
            factor[:, i] = self.factors[np.argmin(diff)][:, i]
        return factor

    # @seed(44)
    @noiser(edge='e4')
    def create_whitener(self):
        """
        Creates whitener.
        """
        return np.einsum('ij,ik->jk', self.m2_eigenvectors_partial,
                         np.diag(1 / np.sqrt(self.m2_eigenvalues_partial)))

    # @seed(45)
    @noiser(edge='e8')
    def create_unwhitener(self):
        """
        Creates unwhitener.
        """
        return np.einsum('ij,ik->jk', self.m2_eigenvectors_partial,
                         np.diag(np.sqrt(self.m2_eigenvalues_partial)))

    # @seed(50)
    @noiser(edge='e9')
    def create_topic_word_distribution(self):
        """
        Creates topic word distribution.
        """
        return self.unwhitener.dot(self.unique_factor).dot(
            np.diag(self.m3_eigenvalues))

    @property
    def n_docs(self):
        return self.document_word_counts.shape[0]

    @property
    def n_words(self):
        return self.document_word_counts.shape[1]

    @property
    def alpha0(self):
        return np.sum(self.alpha)

    @property
    def alpha(self):
        return self.doc_topic_prior

    @property
    def beta(self):
        return self.topic_word_distribution

    @property
    def k(self):
        return self.n_topics

    @property
    def vocab_size(self):
        return self.n_words

    @property
    def l_ns(self):
        return self.document_word_counts.sum(axis=1, keepdims=True)

    @property
    def document_word_proportions(self):
        return self.document_word_counts / \
               self.document_word_counts.sum(axis=1, keepdims=True)

    @property
    def whitened_document(self):
        return self.document_word_counts.dot(self.whitener)

    @property
    def whitened_moment1(self):
        return self.moment1.dot(self.whitener)

    @property
    def m2_eigenvalues_partial(self):
        return self.m2_eigenvalues[:self.n_topics]

    @property
    def m2_eigenvectors_partial(self):
        return self.m2_eigenvectors[:self.n_topics]

