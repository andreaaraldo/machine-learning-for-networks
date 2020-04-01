import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm

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
 

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


def plot_conf_mat(y_true, y_pred, class_names, normalize=True, title=None, 
    cmap=plt.cm.Blues):
 		"""
 		This function prints and plots the confusion matrix.
 		Normalization can be applied by setting `normalize=True`.

 		# Suppose target is the array of the true categories.
		# It contains as many values as the number of samples. Each value is an
		# integer number corresponding to a certain category. This array
		# represents the true category of each sample.
		#
		# predicted has the same format, but it does not represent the true
		# category, rather it represents the result of a model.
		#
		# Note in case of classification models, the categories are the classes,
		# while in case of anomaly detection models, the categories are
		# anomaly / normal
		#
    # In case of errors, you may need to do 
    #     class_names = np.array(class_names)
    # to be sure you satisfy this requirement

 		"""
 		if not isinstance(class_names, (np.ndarray) ):
 		  raise TypeError('class_names must be an np.array. It is instead ', 
                     type(class_names), '. Try to convert to arrays before: executing',
                     'class_names = np.array(class_names)')
    
    
 		if not title:
 		  if normalize:
 		    title = 'Normalized confusion matrix'
 		  else:
 		    title = 'Confusion matrix, without normalization'

 		# Compute confusion matrix
 		cm = confusion_matrix(y_true, y_pred)

 		# Only use the labels that appear in the data
 		labels_present = unique_labels(y_true, y_pred)
 		classes = class_names[labels_present]
 		if normalize:
 			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 			print("Normalized confusion matrix")
 		else:
 			print('Confusion matrix, without normalization')

 		print(cm)

 		fig, ax = plt.subplots(figsize=(8,8))
 		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
 		ax.figure.colorbar(im, ax=ax)
 		# We want to show all ticks...
 		ax.set(xticks=np.arange(cm.shape[1]),
 			yticks=np.arange(cm.shape[0]),
 			# ... and label them with the respective list entries
 			xticklabels=classes, yticklabels=classes,
 			title=title,
 			ylabel='True label',
 			xlabel='Predicted label')

 		# Rotate the tick labels and set their alignment.
 		plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
 			rotation_mode="anchor")

 		# Loop over data dimensions and create text annotations.
 		fmt = '.2f' if normalize else 'd'
 		thresh = cm.max() / 2.
 		for i in range(cm.shape[0]):
 			for j in range(cm.shape[1]):
 				ax.text(j, i, format(cm[i, j], fmt),
 					ha="center", va="center",
 					color="white" if cm[i, j] > thresh else "black")
 		fig.tight_layout()
 		return ax


def plot_feature_importances(importances, feature_names):
  """
  Plots the feature importance with bars. 

  To use with Random Forest Classifiers or Regressors.

  Parameters:
  --------------
  importances: the list of values of feature importances. You can get 
  				it as model.feature_importances (if model is the name of
  				your Random Forest model)
  feature_names: the list of feature names

  Credits to spies006: https://stackoverflow.com/a/44102451/2110769
  """
  indices = np.argsort(importances)
  plt.title('Feature Importances')
  plt.barh(range(len(indices)), importances[indices], color='b', align='center')
  plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
  plt.xlabel('Relative Importance')
  plt.show()




def silhouette_diagram(X_, cluster_labels, n_clusters, title="Silhouette diagram"):
  """
  This code is based on scikit learn documentation:
  https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py


  Returns 
  ----------------
  sample_silhouette_values: the silhouette value of each sample
  """

  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(X_, cluster_labels)

  fig, ax1 = plt.subplots()
  fig.set_size_inches(9, 7)
  ax1.set_xlim([-1, 1])

  # The (n_clusters+1)*10 is for inserting blank space between silhouette
  # plots of individual clusters, to demarcate them clearly.
  ax1.set_ylim([0, len(X_) + (n_clusters + 1) * 10])

  # Compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(X_, cluster_labels)
  y_lower = 10
  for i in range(n_clusters):
  	# Aggregate the silhouette scores for samples belonging to
  	# cluster i, and sort them
  	ith_cluster_silhouette_values = \
  		sample_silhouette_values[cluster_labels == i]
  	ith_cluster_silhouette_values.sort()

  	size_cluster_i = ith_cluster_silhouette_values.shape[0]
  	y_upper = y_lower + size_cluster_i

  	color = cm.nipy_spectral(float(i) / n_clusters)
  	ax1.fill_betweenx(np.arange(y_lower, y_upper),
  		0, ith_cluster_silhouette_values,
  		facecolor=color, edgecolor=color, alpha=0.7)

  	# Label the silhouette plots with their cluster numbers at the middle
  	ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

  	# Compute the new y_lower for next plot
  	y_lower = y_upper + 10  # 10 for the 0 samples

  	ax1.set_title(title)
  	ax1.set_xlabel("The silhouette coefficient values")
  	ax1.set_ylabel("Cluster label")

  	# The vertical line for average silhouette score of all the values
  	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

  return sample_silhouette_values