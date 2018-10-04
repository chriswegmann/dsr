from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

class OpenML:
    def __init__(self, name):
        self.name = name

    def __call__(self):

        dataset = fetch_openml(self.name)
        print(dataset.data.shape)
        print(dataset.target.shape)
        #return dataset.data, dataset.target
        #return dataset.data, LabelEncoder().fit_transform(dataset.target).astype('int64')

        return dataset.data, dataset.target.astype('float32').reshape(-1,1)


        # YOUR CODE HERE:
        #
        # Use scikit-learn's new `fetch_openml` function to download
        # the dataset given by `self.name`.  The function returns a
        # bunch object with two attributes, `dataset.data` and
        # `dataset.target`.  You can print these to see what they look
        # like.  Also print their shape, for example:
        #
        #   print(dataset.data.shape)
        #   print(dataset.target.shape)
        #
        # The `__call__` method that we implement here is supposed to
        # return a tuple of (X, y), where X is a NxM feature matrix
        # and y a target vector of size n.

