import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator


# See https://stackoverflow.com/a/48949667/2110769
# See also https://scikit-learn.org/stable/developers/develop.html


class AndreaLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):

        self.fit_intercept = fit_intercept


    """
    Parameters
    ------------
    column_names: list
            It is an optional value, such that this class knows 
            what is the name of the feature to associate to 
            each column of X. This is useful if you use the method
            summary(), so that it can show the feature name for each
            coefficient
    """ 
    def fit(self, X, y, column_names=() ):

        if self.fit_intercept:
            X = sm.add_constant(X)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        X_cp = X.copy()
        y_cp = y.copy()

        self.X_ = X_cp
        self.y_ = y_cp

        if len(column_names) != 0:
            cols = column_names.copy()
            cols = list(cols)
            X_cp = pd.DataFrame(X_cp)
            cols = column_names.copy()
            cols.insert(0,'intercept')
            X_cp.columns = cols

        self.model_ = sm.OLS(y_cp, X_cp)
        self.results_ = self.model_.fit()
        return self



    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'model_')

        # Input validation
        X_cp = check_array(X.copy() )

        if self.fit_intercept:
            X_cp = sm.add_constant(X_cp)
        return self.results_.predict(X_cp)
        

    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept}


    def summary(self):
        print(self.results_.summary() )



def subsample_two_arrays(X, y, n,random_state=None):
	"""
	Take a subsample of X and y composed of n rows.

	Parameters
	-------------------
	X:		Array
	y:		Array. It must have the same rows as X
	n:		Number of rows to take
	random_state: 	the seed


	Returns
	-------------------
	X_sub:	A sample of n rows from X, selected randomly
	y_sub:	A sample of n rows from y. The same row ids of X_sub are 
	 		selected
	"""

	if X.shape[0] != y.shape[0]:
		raise ValueError("X and y must have the same rows")

	
	X_sub, X_ignore, y_sub, y_ignore = \
				train_test_split(X,y, train_size=n , 
					random_state=random_state)

	return X_sub, y_sub


def subsample(X, n,random_state=None):
	"""
	Take a subsample of X and y composed of n rows.

	Parameters
	-------------------
	X:		Array
	n:		Number of rows to take
	random_state: 	the seed


	Returns
	-------------------
	X_sub:	A sample of n rows from X, selected randomly
	"""
	
	X_sub, X_ignore = train_test_split(X, train_size=n , 
					random_state=random_state)

	return X_sub