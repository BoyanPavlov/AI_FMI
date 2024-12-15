import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Fit the Naive Bayes model.
        X: numpy array, shape (n_samples, n_features)
        y: numpy array, shape (n_samples,)
        """
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _calculate_likelihood(self, x, mean, var):
        """Calculate the Gaussian likelihood."""
        eps = 1e-6  # Small value to prevent division by zero
        coeff = 1.0 / np.sqrt(2 * np.pi * (var + eps))
        exponent = -((x - mean) ** 2) / (2 * (var + eps))
        return coeff * np.exp(exponent)

    def _calculate_posterior(self, x):
        """Calculate posterior probability for each class."""
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihoods = np.sum(np.log(self._calculate_likelihood(x, self.mean[c], self.var[c])))
            posteriors[c] = prior + likelihoods
        return posteriors

    def predict(self, X):
        """Predict the class for each sample in X."""
        predictions = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Example Usage
if __name__ == "__main__":
    # Dummy dataset
    X = np.array([[1.0, 2.1], [1.2, 1.9], [2.0, 2.2], [2.5, 2.8], [3.0, 3.2]])
    y = np.array([0, 0, 1, 1, 1])

    # Train the classifier
    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    # Test prediction
    X_test = np.array([[1.5, 2.0], [3.0, 3.1]])
    predictions = clf.predict(X_test)

    print("Predictions:", predictions)
