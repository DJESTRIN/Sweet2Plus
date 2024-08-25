<h1> <b> 🔬 Sweet2Plus 🔬 </b> </h1> 
A python API for analysis of suite2P outputs. In addition to parsing suite2P outputs, this repo includes various statistical analyses in python and R. This repo also contains a reliable neural network (MLP) 🧠 for signal classification. 
We include scripts allowing for easy plug in of DeepLabCut based pose-estimation results. This repository contains C++ code for running behavioral experiments in the wet lab, as well as a guided user interface for managing multiple ports
from various behavioral computers simultaneously.

<h2> <b> ⚠️ Warning: This code is still under development. ⚠️ </b> </h2>
Please kindly ignore any issues with code as well as any missing citations to others code. 

<h2> <b> Sweet2Plus's API </b></h2>
Sweet2Plus allows for the analysis of two-photon calcium imaging data. Here are a few example images from our dataset:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/example1.png" width="300" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/example2.png" width="300" /> 
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/example3.png" width="300" /> 
</p>

Although there is more to come, we utalize code to functionally define neuronal cell types and then perform analyses based on these cell types. 
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/kmeans_clustering.png" width="500" />
</p>

<h2> <b> Example Neural Network Classifier Performance </b></h2>
We trained a Pytorch based Neural Network (multi-layer perceptron) to parse Suite2P suggested regions of interest (ROIs) into real vs not-real signal. In addition to training, our model was enhanced through Optuna (bayesian optimization) hyper-parameter tuning. 

First, traces are read into python and normalized to a pre-determined number of points.
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/exampletrace.png" width="500" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/examplenormtrace.png" width="500" /> 
</p>

Overall, the classifier performed moderatly well with a test F1 score >0.82:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/MLP_resultsds.png" width="500" />
</p>

The classifier was optimized via Bayesian Optimization available with the Optuna library:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/newplot.png" width="500" />
</p>

Now, we quickly use this classifier to parse real from not-real ROIs suggested by Suite2P. Notably, not-real ROIs include ROIs containing no-signal or extremly low signal with S:N < 3:1. 
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/RealSignal.png" width="500" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/images/Noise.png" width="500" /> 
</p>


<h2> <b> References </b></h2>
Portions of this library utalize code from (or are inspired by) the following references:

- <b> Two-Photon Calcium Imaging: </b> Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L. F., Dalgleish, H., Carandini, M., & Harris, K. D. (2017). Suite2p: Beyond 10,000 neurons with standard two-photon microscopy. bioRxiv, 061507. https://doi.org/10.1101/061507

- <b> Pose-Estimation: </b> Mathis, A., Mamidanna, P., Cury, K. M., Abe, T., Murthy, V. N., Mathis, M. W., & Bethge, M. (2018). DeepLabCut: Markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience, 21(9), 1281-1289. https://doi.org/10.1038/s41593-018-0209-y
  
- <b> Bayesian Optimization: </b>  Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. arXiv preprint arXiv:1907.10902. https://arxiv.org/abs/1907.10902


<h2> <b> Contributions and citation </b> </h2>
David James Estrin & Kenneth Wayne Johnson contributed equally on this project.

- Code: David James Estrin, Kenneth Wayne Johnson
  
- Data: Kenneth Wayne Johnson, David James Estrin

Please cite this git repository as Estrin, D.J., Johnson, K. W. et al., (2025) Sweet2Plus: A python API for analysis of suite2P outputs. unpublished if you use any code or intellectual property from it. Thank you!


