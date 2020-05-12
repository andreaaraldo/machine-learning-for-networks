import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score,\
				average_precision_score, precision_score, recall_score, \
				precision_recall_curve, roc_curve, roc_auc_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

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
 


def plot_conf_mat(y_true, y_pred, class_names, normalize=True, title=None, 
    cmap=plt.cm.Blues, text=True, width=8, height=8 ):
 		"""
 		This function prints and plots the confusion matrix.
 		In case of errors, you may need to do 
 				class_names = np.array(class_names)
 		before calling this function.


 		Parameters:
 		--------------------------
		target: The array of the true categories. It contains as many values 
				as the number of samples. Each value is an integer number 
				corresponding to a certain category. This array represents 
				the true category of each sample.
		
		predicted:	It has the same format, but it does not represent the true 
					category, rather it represents the result of a model.

		class_names:	Array of strings, where the first. The k-th element
						is the name of the k-th class

		normalize: 	(default=True) If False, it just prints the number of values in 
						each cell. Otherwise it prints the frequencies, i.e.
						the sum over each row is 1

		title: 	(default=None) Title of the figure

		cmap: 	(default=plt.cm.Blues) Color map

		text:	(default=True) If True it prints numerical values on each cell. Otherwise
				it just shows the colors


		width: 	(default=8) Of the figure

		height:	(default=8) Of the figure
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

 		fig, ax = plt.subplots(figsize=(width,height))
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
 		if text == True:
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




def silhouette_diagram(X_, cluster_labels, n_clusters, sample_size=None,
	random_state = None,
	title="Silhouette diagram"):
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




###########################################
###### ANOMALY DETECTION ##################
###########################################


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp/(fp+tn)
    return fpr

def plot_precision_recall_curve(y_train, anomaly_scores, ax, 
                                threshold_selected=None):
  """
  Inspired by 
    http://abhay.harpale.net/blog/machine-learning/threshold-tuning-using-roc/
  """
  
  precision, recall, thresholds = precision_recall_curve(y_train, anomaly_scores)
  pr_auc_score = average_precision_score(y_train,anomaly_scores)

  ax.plot(recall, precision, 
           label='Pr-Re curve (AUC = %0.2f)' % (pr_auc_score))
  ax.set_ylabel('Precision')
  ax.set_xlabel('Recall')

  ax_tw = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax_tw.plot(recall[:-1], thresholds,linestyle='dashed', color='grey', 
              label='Threshold')
  ax_tw.tick_params(axis='y')
  ax_tw.set_ylabel('Threshold')
  ax.grid(True)
  ax.legend(loc="lower center")
  ax_tw.legend(loc="lower left")
  ax.set_xticks(np.arange(0, 1+0.09, 0.1))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))

  if threshold_selected != None:
    y_pred = (anomaly_scores >= threshold_selected)
    recall_sc = recall_score(y_train, y_pred)
    print("Precision=", precision_score(y_train, y_pred))
    print("Recall=", recall_sc)
    ax.axvline(recall_sc, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")
    


def plot_roc_curve(y_train, anomaly_scores, ax, threshold_selected=None):
  """
  Inspired by 
    http://abhay.harpale.net/blog/machine-learning/threshold-tuning-using-roc/
  """
  
  fpr, tpr, thresholds = roc_curve(y_train, anomaly_scores)
  roc_auc_sc = roc_auc_score(y_train, anomaly_scores)
  # The thresholds obtained here are the same as the ones obtained with the
  # precision-recall curve
  
  ax.plot(fpr, tpr, label='ROC (AUC = %0.2f)' % (roc_auc_sc))
  ax.set_ylabel('True Positive Rate (Recall) : TP/(TP+FN) )')
  ax.set_xlabel('False Positive Rate: FP/(FP+TN)')

  # Also plot the thresholds
  ax_tw = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax_tw.plot(fpr, thresholds,linestyle='dashed', color='grey', 
              label='Threshold')
  ax_tw.tick_params(axis='y')
  ax_tw.set_ylabel('Threshold')
  ax.grid(True)
  ax.legend(loc="center right")
  ax_tw.legend(loc="lower left")
  ax.set_xticks(np.arange(0, 1+0.09, 0.1))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))


  if threshold_selected != None:
    y_pred = (anomaly_scores >= threshold_selected)
    tpr = recall_score(y_train, y_pred)
    fpr = false_positive_rate(y_train, y_pred)

    print("False Positive Rate = ", fpr)
    print("True Positive Rate = ", tpr)
    ax.axvline(fpr, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")




def plot_precision_recall_vs_thresholds(y_train, anomaly_scores, ax, 
                                        threshold_selected=None):
  """

    From Fig.3.4 of Geron, Hands-On Machine Learning with Scikit-Learn, 
      Keras, and TensorFlow, O'Reilly 2019
  """
  precision, recall, thresholds = precision_recall_curve(y_train, anomaly_scores)
  ax.plot(thresholds, precision[:-1], 'b--', label='Precision')
  ax.plot(thresholds, recall[:-1], 'g-', label='Recall')
  ax.set_xlabel('Threshold')
  ax.legend(loc='upper center')
  ax.grid(True)
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))
  
  if threshold_selected != None:
    ax.axvline(threshold_selected, 0, 1, color='r', 
      linestyle='-.', label="Threshold selected")




def plot_tpr_fpr_vs_thresholds(y_train, anomaly_scores, ax, 
                               threshold_selected=None):
  """
  
    Inspired by Fig.3.4 of Geron, Hands-On Machine Learning with Scikit-Learn, 
      Keras, and TensorFlow, O'Reilly 2019
  """
  fpr, tpr, thresholds = roc_curve(y_train, anomaly_scores)
  ax.plot(thresholds, fpr, 'b--', label='False Positive rate')
  ax.plot(thresholds, tpr, 'g-', label='True Positive rate')
  ax.set_xlabel('Threshold')
  ax.legend(loc='upper center')
  ax.grid(True)
  #ax.set_xticks(np.arange(min(thresholds),max(thresholds), 
  #                       (max(thresholds) -min(thresholds) )/10  ))
  #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  ax.set_yticks(np.arange(0, 1+0.09, 0.1))
  
  if threshold_selected != None:
    ax.axvline(threshold_selected, 0, 1, color='r', 
               linestyle='-.', label="Threshold selected")



def evaluate_anomaly_detector(y_train, anomaly_scores, threshold_selected=None):
  """
    Evaluates the results from anomaly detection
  """
  fig, axs = plt.subplots(2,2, figsize=(16, 16))
  fig.tight_layout(pad=8)
  plt.rcParams["font.size"] = "12"

  plot_precision_recall_curve(y_train, anomaly_scores, axs[0,0], threshold_selected)
  plot_roc_curve(y_train, anomaly_scores, axs[0,1], threshold_selected)  
  plot_precision_recall_vs_thresholds(y_train, anomaly_scores, axs[1,0], 
                                      threshold_selected)
  plot_tpr_fpr_vs_thresholds(y_train, anomaly_scores, axs[1,1], 
                             threshold_selected)
  

