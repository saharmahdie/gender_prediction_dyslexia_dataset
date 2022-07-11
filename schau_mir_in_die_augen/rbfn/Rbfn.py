"""Radial Basis Function Networks"""

# Christoph Schröder <schroeder.c@cs.uni-bremen.de> 2019
import scipy
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class Rbfn(BaseEstimator):
    """Radial Basis Function Networks

    Based on George, Anjith, and Aurobinda Routray. “A Score Level Fusion Method for Eye Movement Biometrics.”
    Pattern Recognition Letters 82 (2016): 207–15. https://doi.org/10.1016/j.patrec.2015.11.020.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form.
    use_cuda : bool, default: False
        Calculate the pseudo inverse with cude. This currently is not working as the result is wrong.
    """
    def __init__(self, n_clusters=8, use_cuda=False, random_state=42):
        # hyperparameter
        self.random_state = random_state
        self.n_clusters = n_clusters

        # learned parameters
        self.w = np.array([])
        self.mus = np.array([])
        self.betas = np.array([])

        self.classes_ = []

        # we need scaled data and classes one-hot encoded
        self.enc = OneHotEncoder(categories='auto', sparse=False)
        self.scaler = StandardScaler()

        # config
        self.use_cuda = use_cuda

    def fit(self, x, y):
        """ Train the classifier.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.
        """
        x, y = check_X_y(x, y, ensure_min_samples=2, estimator=self)
        self.classes_ = unique_labels(y)
        n_samples, _ = x.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")

        # 0. scale each feature to zero center, norm variance
        X_unscaled = x
        x = self.scaler.fit_transform(x)

        # allocate space for our parameters
        self.mus = np.zeros((self.n_clusters * n_classes, x.shape[1]))
        self.betas = np.zeros(len(self.mus))

        # 1. find n_clusters support vectors
        # calculates all \mu s with k-means:
        #   - 1 nearest neighbor
        #   - euclidean distance
        #   - max 100 iterations
        simple_clustering = False
        if simple_clustering:
            kmns = KMeans(n_clusters=self.n_clusters * n_classes, max_iter=100, random_state=self.random_state, n_jobs=4)
            kmns.fit(x)
            self.mus = kmns.cluster_centers_
            for j, c in enumerate(self.mus):
                cluster_samples = x[kmns.labels_ == j]
                sigma = np.mean(distance.cdist(cluster_samples, np.asarray([c])))
                # add small delta to avoid division by zero
                self.betas[j] = 1 / (2 * (sigma+0.00001)**2)
        else:
            mu_count = 0
            # estimate \mu and \beta for each of the self.n_clusters * n_classes neurons
            for i, c  in enumerate(self.classes_):
                # range to save our parameters in
                param_slice = slice(i * self.n_clusters, (i + 1) * self.n_clusters)

                # subset with only training data for the current class
                X_subset = x[y == c, :]

                # mus for the current class
                kmns = KMeans(n_clusters=self.n_clusters, max_iter=100, random_state=self.random_state, n_jobs=4)
                kmns.fit(X_subset)
                self.mus[param_slice] = kmns.cluster_centers_

                # 2. estimate n_clusters \betas
                # \beta = 1/(2*\sigma^2)
                # \sigma = mean distance between cluster center and all of it's elements
                for j, c in enumerate(kmns.cluster_centers_):
                    # cluster_samples = x[kmns.labels_ == j]
                    cluster_samples = X_subset[kmns.labels_ == j]
                    # cluster_size = len(cluster_samples)
                    # if cluster_size < 3:
                    #     print("small cluster: {}".format(cluster_size))
                    sigma = np.mean(distance.cdist(cluster_samples, np.asarray([c])))
                    # add small delta to avoid division by zero
                    self.betas[mu_count] = 1 / (2 * (sigma+0.00001)**2)
                    mu_count += 1

        # 3. find n_classes weigths (why not n_classes * self.n_clusters?)
        # input all data and find w using the Moore–Penrose pseudoinverse
        # A = f_{i,j}(x_k)
        # i = 1,...,self.n_clusters (K)
        # j = 1,...,n_classes (m)
        # k = 1,...,n_samples (n = m*c)
        # Aŵ=ŷ
        A = self._activate(x)

        # if we divide by 0 in the beta calculation we get nans, replace them by 0
        nan_count = np.sum(np.isnan(A))
        if nan_count > 0:
            print("replacing zeros in activation".format(nan_count))
            A[np.isnan(A)] = 0

        print("Calculating pinv of size {}".format(A.shape))
        if self.use_cuda:
            print("WARNING: CUDA currently gives the wrong result!")
            print("Using cuda")
            import pycuda.gpuarray as gpuarray
            import pycuda.autoinit # import is needed
            import skcuda.linalg as linalg
            linalg.init()
            print("Moving A to GPU")
            a_gpu = gpuarray.to_gpu(np.asarray(A.T, np.float32))
            print("Calculating pinv")
            a_inv_gpu = linalg.pinv(a_gpu)
            print("Moving A back to CPU")
            Ap = a_inv_gpu.get().T
        else:
            #Ap = scipy.linalg.pinv(A)
            Ap = np.linalg.pinv(A)

        print("Calculating w")
        # y in one hot encoding
        y_hot = self.enc.fit_transform(y.reshape(-1, 1))
        self.w = Ap @ y_hot

        # test on training data
        print("\tTraining accuracy: %1.3f" % accuracy_score(y, self.predict(X_unscaled)))
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
        from scipy.special import softmax
        prediction_normalized = softmax(prediction, axis=1)  ## to have prbablties between 0 and 1
        # print("prediction_normalized here", prediction_normalized)

        return prediction_normalized

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



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
#
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# clf = Rbfn(n_clusters=13) #RandomForestClassifier()#
# clf.fit(X_train, y_train)
#
#
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))
