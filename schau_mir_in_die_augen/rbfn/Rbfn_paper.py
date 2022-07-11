"""Radial Basis Function Networks"""

# Christoph Schröder <schroeder.c@cs.uni-bremen.de> 2019
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Rbfn(BaseEstimator):
    """Radial Basis Function Networks

    Based on George, Anjith, and Aurobinda Routray. “A Score Level Fusion Method for Eye Movement Biometrics.”
    Pattern Recognition Letters 82 (2016): 207–15. https://doi.org/10.1016/j.patrec.2015.11.020.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form.
    """
    def __init__(self, n_clusters=8):
        # hyperparameter
        self.n_clusters = n_clusters

        # learned parameters
        self.w = np.array([])
        self.mus = np.array([])
        self.betas = np.array([])

        self.classes_ = []

        # we need scaled data and classes one-hot encoded
        self.enc = OneHotEncoder(categories='auto', sparse=False)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """ Train the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")

        # 0. scale each feature to zero center, norm variance
        X = self.scaler.fit_transform(X)

        # allocate space for our parameters
        self.mus = np.zeros((self.n_clusters * n_classes, X.shape[1]))
        self.betas = np.zeros(len(self.mus))

        # 1. find n_clusters support vectors
        # calculates all \mu s with k-means:
        #   - 1 nearest neighbor
        #   - euclidean distance
        #   - max 100 iterations
        mu_count = 0
        # estimate \mu and \beta for each of the self.n_clusters * n_classes neurons
        for i in range(n_classes):
            # range to save our parameters in
            param_slice = slice(i * self.n_clusters, (i + 1) * self.n_clusters)

            # subset with only training data for the current class
            X_subset = X[y==i, :]

            # mus for the current class
            kmns = KMeans(n_clusters=self.n_clusters, max_iter=100, random_state=i, n_jobs=4)
            kmns.fit(X_subset)
            self.mus[param_slice] = kmns.cluster_centers_

            # 2. estimate n_clusters \betas
            # \beta = 1/(2*\sigma^2)
            # \sigma = mean distance between cluster center and all of it's elements
            for j, c in enumerate(kmns.cluster_centers_):
                cluster_samples = X_subset[kmns.labels_ == j]
                sigma = np.mean(distance.cdist(cluster_samples, np.asarray([c])))
                # add small delta to avoid division by zero
                self.betas[mu_count] = 1 / (2 * (sigma+0.0000001)**2)
                mu_count += 1

        # 3. find n_classes weigths (why not n_classes * self.n_clusters?)
        # input all data and find w using the Moore–Penrose pseudoinverse
        # A = f_{i,j}(x_k)
        # i = 1,...,self.n_clusters (K)
        # j = 1,...,n_classes (m)
        # k = 1,...,n_samples (n = m*c)
        # Aŵ=ŷ
        A = self._activate(X)

        print("Calculating pinv of size {}".format(A.shape))
        Ap = np.linalg.pinv(A)

        print("Calculating w")
        # y in one hot encoding
        y_hot = self.enc.fit_transform(y.reshape(-1, 1))
        self.w = Ap @ y_hot

        return self

    def predict(self, X):
        """Predict the class each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Class each sample belongs to.
        """

        return self.enc.inverse_transform(self.predict_proba(X))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        X = self.scaler.transform(X)
        activation = self._activate(X)
        prediction = activation @ self.w

        return prediction

    def _activate(self, X):
        """ Calculate activation vector for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        :return: ndarray
            array, shape [n_samples, n_classes * n_clusters]
        """
        activations = []
        for f in X:
            # Eq. 3 from the paper:
            # f is broadcasted to len(mus)
            d = f - self.mus
            # squared norm
            res = np.einsum('ij,ij->i',d,d)
            activations.append(np.exp(-self.betas * res))

        return np.asarray(activations)


from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = Rbfn(n_clusters=13)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))
