# Machine Learning for high-speed networks.


## Intro

In this guided class, we will realize the components of a high-speed network ML application.
In the first part, you are going to train a network to predict what is the network load, based on some features.
In the second part, you are going to use the trained model in an emulated scenario to make predictions at line rate.

## Training and Validation

* Objective: Train a NN model that is able to predict the traffic load (label at the last column) using the given features (n-1 first columns).
Store the trained model in a file. Load the trained model in a simple test scenario

* Dataset
The dataset contains some features related to CPU measurements in a network scenario.
The last column of the dataset is the label. The 3 label refer to the three cases:
- 0: low load
- 1: mid load
- 2: high load


All the datasets are in .csv format
The interesting datasets are :
- fulldataset.csv
- highdataset.csv
- middataset.csv
- lowdataset.csv

Check the features with the command:

``` 
head fulldataset.csv

> time,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,branch-load-misses,branch-misses,branches,bus-cycles,cache-misses,cache-references,context-switches,cpu-clock,cycles,dTLB-load-misses,dTLB-store-misses,dTLB-stores,iTLB-load-misses,iTLB-loads,instructions,minor-faults,node-load-misses,node-loads,node-store-misses,node-stores,page-faults,ref-cycles,task-clock,label

```
While the full dataset includes all samples, the high/low/mid csv files include only one specific class.

The notebook includes a skeleton of the typical steps for doing such a task.


## Testbed emulation

* Objective: Load the trained model and perform some predictions in the RX function. 
Check if there are tradeoffs between the accuracy of the model and the traffic rate.
Play with the parameters of the emulation (rate, queue size, per-packet classification vs sampling).

* Structure

1) txgen function
It reads some data from a .csv file and writes the data on a shared queue. No modifications are needed here.

2) rx function
It emulates the RX of a typical high-speed router. After packets are read, it calls the **processing** function, which is the most important function.

3) processing function
After the model is loaded, this function should be able to give predictions and return a value.
Here is the portion to modify.

4) main
At the end of the main function, you will see how many packets are sent, received and lost.

* Parameters

shared = Queue(maxsize=1024)   # Max size of the queue: can be modified to force additional losses
rate = .1 # Rate for the TX generation: should be decreased for a higher rate
duration = 5 # Number of seconds of the emulation


