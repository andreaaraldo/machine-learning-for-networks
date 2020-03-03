from sklearn.feature_selection import VarianceThreshold
import numpy as np


def low_var_features(df, threshold):
	    """ 
	    Returns the features whose variance is 
	    above the threshold
	    
	    Parameters
	    ------------
	    df: dataframe
	    threshold : float
	    
	    Returns
	    -----------
	    list
	    	the names of the features whose variance is below
	    	the threshold
	    """
	    feature_filter = VarianceThreshold(threshold=threshold)
	    feature_filter.fit(df);
	    features_above_thres_idx = \
	    df.columns[feature_filter.get_support()];
	    features_above_thres = list(features_above_thres_idx);
	    return features_above_thres;


def get_most_correlated(df):
	"""
	Returns the pairs of features in descending order of Person's 
	correlation

	Parameters
	--------------
	df: dataframe

	Returns
	-------------
	Series
		the pair of features and the Pearson's coefficient
	"""

	corrmatrix = df.corr().round(3)


	# Trick from https://stackoverflow.com/a/43073761/2110769
	#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
	highest_correlated = (corrmatrix.where(np.triu(np.ones(corrmatrix.shape), k=1).astype(np.bool))
	                 .stack()
	                 )
	highest_correlated =\
			highest_correlated.reindex(highest_correlated.abs().
						sort_values(ascending=False).index)

	#first element of sol series is the pair with the largest correlation

	return highest_correlated

def get_features_correlated_to_target(df, target_feature):
	"""
	Returns the Pearson's correlation coefficient between the 
	features and the target

	Parameters
	------------
	df_: dataframe

	target_feature: string
		the name of the target feature

	Returns
	--------------
	Series
		A series of features, with the Person's correlation 
		coefficient between them and the target, ordered in decreasing
		oder of correlation
	"""
	corrmatrix = df.corr().round(2)
	corr_to_label_num = corrmatrix[target_feature]
	# We sort based on the absolute value (see https://stackoverflow.com/a/30486411/2110769)
	features_corr = corr_to_label_num.reindex(corr_to_label_num.abs().
		sort_values(ascending=False).index)
	return features_corr;