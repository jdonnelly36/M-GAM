import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class MAR_model():
    '''
    A convenience class for defining MAR models
    '''
    def __init__(self, weights, missingness_prob=0.2):
        '''
        Parameters
        ----------
        weights : list of vectors (num_features)
            The weights to use in a logistic regression
            model to decide whether each feature will be
            obfuscated for a given sample
        missingness_prob: float
            The proportion of samples that should be obfuscated
            for each feature
        '''
        self.weights = weights
        self.missingness_prob = missingness_prob

    def get_mask(self, X):
        '''
        This function gets a binary mask indicating
        whether each entry in the matrix X should be 
        obfuscated according to the given parameters
        
        Parameters
        ----------
        X : matrix (num_samples, num_features)
            The original unperturbed input features

        Returns
        -------
        mask : array (num_samples, num_features)
            A binary mask indicating whether each
            entry in X should be obfuscated

        '''
        mask = np.zeros_like(X)

        # For each feature in X, get the probability of removal
        for ftr_ind, w in enumerate(self.weights):
            unajusted_predictions = sigmoid(X @ w)

            # Take the given predictions and create a mask such that missingness_prob
            # of the entries are missing
            positive_threshold = np.quantile(unajusted_predictions, 1 - self.missingness_prob)
            cur_mask = np.where(unajusted_predictions.flatten() > positive_threshold, 0, 1)
            mask[:, ftr_ind] = cur_mask
        return mask

