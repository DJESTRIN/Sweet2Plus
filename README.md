<h1> <b> 🔬 Sweet2Plus 🔬 </b> </h1> 
A python API for analysis of suite2P outputs. In addition to parsing suite2P outputs, this repo includes various statistical analyses in python and R. This repo also contains a reliable neural network (MLP) 🧠 for signal classification. 
We include scripts allowing for easy plug in of DeepLabCut based pose-estimation results. This repository contains C++ code for running behavioral experiments in the wet lab, as well as a guided user interface for managing multiple ports
from various behavioral computers simultaneously.

<h2> <b> ⚠️ Warning: This code is still under development. ⚠️ </b> </h2>
Please kindly ignore any issues with code as well as any missing citations to others code. 

<h2> <b> Sweet2Plus's API </b></h2>
Sweet2Plus allows for the analysis of two-photon calcium imaging data. Here are a few example images from our dataset:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/tree/main/images/example1.png" width="300" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/tree/main/images/example2.png" width="300" /> 
  <img src="https://github.com/DJESTRIN/Sweet2Plus/tree/main/images/example3.png" width="300" /> 
</p>

 
![https://github.com/DJESTRIN/BrainBeam/tree/lightsheet_cluster/BrainBeam/gui/gui_images/gui_08022024.png?raw=True](https://github.com/DJESTRIN/BrainBeam/blob/lightsheet_cluster/BrainBeam/gui/gui_images/gui_08022024.png)

<h2> <b> Example Classifier Performance </b></h2>
We trained an ilastik based classifier to quantify cell counts for whole-brain light-sheet data. In addition to training, models are enhanced through hyper-parameter tuning:
<p float="center">
  <img src="https://github.com/DJESTRIN/BrainBeam/blob/lightsheet_cluster/BrainBeam/gui/gui_images/hyperparameter_tuning.png" width="500" />
</p>

The overall performance of our models results in an F1 score >0.85:
<p float="center">
  <img src="https://github.com/DJESTRIN/BrainBeam/blob/lightsheet_cluster/BrainBeam/gui/gui_images/classifier_performance.png" width="500" />
</p>

Here is an example of our ilastik classifier's performance on a sagittal slice of brain tissue:
<p float="left">
  <img src="https://github.com/DJESTRIN/BrainBeam/blob/lightsheet_cluster/BrainBeam/gui/gui_images/sagittal1.png" width="500" />
  <img src="https://github.com/DJESTRIN/BrainBeam/blob/lightsheet_cluster/BrainBeam/gui/gui_images/sagittal1_labeled.png" width="500" /> 
</p>


<h2> <b> References </b></h2>
Portions of this library utalize code from (or are inspired by) the following references:

- <b> De-striping: </b> Kirst, et al. (2020). Mapping the fine-scale organization and plasticity of the brain vasculature. Cell, 180(4), 780-795. https://doi.org/10.1016/j.cell.2020.01.028

- <b> De-striping: </b> Renier et al. (2016). Mapping of brain activity by automated volume analysis of immediate early genes. Cell, 165(7), 1789-1802. https://doi.org/10.1016/j.cell.2016.05.007

- <b> Stitching: </b> Bria, A., & Iannello, G. (2012). TeraStitcher - A tool for fast automatic 3D-stitching of teravoxel-sized microscopy images. BMC Bioinformatics, 13(1), 316. https://doi.org/10.1186/1471-2105-13-316

- <b> Cell Segmentation: </b> Athey, T. L., Wright, M. A., Pavlovic, M., Chandrashekhar, V., Deisseroth, K., Miller, M. I., & Vogelstein, J. T. (2023). BrainLine: An open pipeline for connectivity analysis of heterogeneous whole-brain fluorescence volumes. Neuroinformatics, 21(4), 637-639. https://doi.org/10.1007/s12021-023-09638-2

- <b> Cell Segmentation: </b> Berg, S., Kutra, D., Kroeger, T., Straehle, C. N., Kausler, B. X., Haubold, C., Schiegg, M., Ales, J., Beier, T., Rudy, M., Eren, K., Cervantes, J. I., Xu, B., Beuttenmueller, F., Wolny, A., Zhang, C., Koethe, U., Hamprecht, F. A., & Kreshuk, A. (2019). ilastik: Interactive machine learning for (bio)image analysis. Nature Methods. https://doi.org/10.1038/s41592-019-0582-9

- <b> Brain Registration: </b> Chandrashekhar, V., Tward, D. J., Crowley, D., et al. (2021). CloudReg: Automatic terabyte-scale cross-modal brain volume registration. Nature Methods, 18(8), 845–846. https://doi.org/10.1038/s41592-021-01218-z

<h2> <b> Contributions and citation </b> </h2>

- Code: David James Estrin 

- Data: David James Estrin, Christine Kuang

Please cite this git repository as Estrin, D.J., et al., (2025) BrainBeam: A generalized open-source pipeline and gui for analyzing light sheet brain tissue. unpublished if you use any code or intellectual property from it. Thank you!


