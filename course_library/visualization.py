import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def plot_corr(df, width, height, print_value, thresh=0):
	    """ Plot a correlation plot
	    
	    Parameters
	    ------------
	    df: dataframe
			The dataframe of which we want to show 
			the correlation
	    width : int
	    	The width of the figure
	    height : int
	    	The height of the figure	    
	    print_value : bool
	    	If True, it prints the Pearson's 
	    	correlation coefficient values in 
	    	the picture
	    thresh : float
	    	No color will be shown if the correlation 
	    	value is below this threshold 
	    	in absolute value
	    """
	    cormat = df.corr()
	    cormat = cormat.where(abs(cormat)>=thresh)
	    cormat = cormat.fillna(0)
	    
	    mask = np.zeros_like(cormat, dtype=np.bool)
	    mask[np.triu_indices_from(mask)] = True
	    
	    # Inspired by https://stackoverflow.com/a/42977946/2110769
	    f, ax = plt.subplots(figsize=(width, height ) )
	    
	    # Inspired by https://medium.com/@chrisshaw982/seaborn-correlation-heatmaps-customized-10246f4f7f4b
	    sns.heatmap(cormat,
		    vmin=-1,
		    vmax=1,
		    cmap='coolwarm',
		    annot=print_value,
		       mask = mask);



def rotate_labels(sm):
		"""
		Rotate the labels of a scatter matrix

		Parameters
		-----------------
		sm: scatter matrix
		"""
		# source: https://stackoverflow.com/a/32568134/2110769
		[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
		[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
 