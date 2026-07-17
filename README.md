<h1> <b> 🔬 Sweet2Plus 🔬 </b> </h1> 
A python API for analysis of suite2P outputs. In addition to parsing suite2P outputs, this repo includes various statistical analyses in python and R. This repo also contains a reliable neural network (MLP) 🧠 for signal classification. 
We include scripts allowing for easy plug in of DeepLabCut based pose-estimation results. This repository contains C++ code for running behavioral experiments in the wet lab, as well as a guided user interface for managing multiple ports
from various behavioral computers simultaneously.

<h2> <b> Table of Contents </b></h2>

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running the test suite](#running-the-test-suite)
- [Sweet2Plus's API](#sweet2pluss-api)
- [Example Neural Network Classifier Performance](#example-neural-network-classifier-performance)
- [References](#references)
- [Contributions and citation](#contributions-and-citation)

<h2> <b> Repository layout </b></h2>

```
Sweet2Plus/
├── Sweet2Plus/                 # the installable Python package
│   ├── core/                   suite2p wrapper, behavior parsing, object serialization
│   ├── analysis/                correlative-activity and stimulation analyses
│   ├── statistics/              GLMs, mixed models, clustering, and heatmap statistics (Python + R helpers)
│   ├── pose_estimation/          DeepLabCut output parsing utilities
│   ├── weight_modeling/          modeling of synaptic weight dynamics
│   ├── graphics/                figure/plotting helpers
│   ├── decoders/                neural network decoders (MLP, GNN, autoencoders)
│   ├── signalclassifier/        MLP-based real vs. noise ROI classifier
│   ├── denoise/                 DeepCAD-based denoising and destriping utilities
│   ├── utils/                   logging, parallelization, and misc helpers
│   └── cluster_scripts/         cluster/HPC job scripts (was cloud/)
├── portreader_gui/              GUI for managing multiple behavioral-computer serial ports
├── firmware/arduino/            C++ firmware for running behavioral experiments (was arduino/)
├── docs/images/                 README figures (was images/)
├── tests/                       pytest smoke-test suite (see below)
├── Dockerfile / requirements.txt  reproducible test/analysis environment
└── setup.py                     package metadata (installable via pip)
```

<h2> <b> Installation </b></h2>

Sweet2Plus targets Python 3.10+. From the repository root:

```bash
# Editable install with the package's Python dependencies
pip install -r requirements.txt
pip install -e .
```

If you use conda, create an environment first (e.g. `conda create -n sweet2plus python=3.10 && conda activate sweet2plus`) and then run the commands above inside it.

Alternatively, `environment-suite2p310.yml` is a full `conda env export` snapshot
(Windows, Python 3.10) of a known-working environment for the suite2p-based
pipeline; create it with `conda env create -f environment-suite2p310.yml`.
This is a supplementary, platform-specific reference — prefer
`requirements.txt` + `pip install -e .` for normal development.

A few dependencies are intentionally **not** installed automatically:
- `deepcad` (used only by `Sweet2Plus.denoise.RunDeepCAD`) is not published on PyPI under a matching name — the real [DeepCAD-RT](https://github.com/cabooster/DeepCAD-RT) project must be installed manually from GitHub if you need this module.
- `projectmanager` (used only by `Sweet2Plus.utils.logger`'s `s2p_logger`/`watch_directory` helpers) is a private, unpublished package; the rest of the logger works fine without it.

<h2> <b> Running the test suite </b></h2>

The repo ships a Docker image that mirrors the tested environment, plus a pytest smoke-test suite that imports every module to catch packaging/import regressions.

```bash
# Build the image
docker build -t sweet2plus .

# Run the test suite
docker run --rm sweet2plus

# Or drop into a shell for interactive debugging
docker run --rm -it sweet2plus bash
```

You can also run pytest directly against a local environment that has `requirements.txt` installed:

```bash
pytest -v
```

<h2> <b> Sweet2Plus's API </b></h2>
Sweet2Plus allows for the analysis of two-photon calcium imaging data. Here are a few example images from our dataset:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/example1.png" width="300" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/example2.png" width="300" /> 
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/example3.png" width="300" /> 
</p>

Although there is more to come, we utalize code to functionally define neuronal cell types and then perform analyses based on these cell types. 
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/kmeans_clustering.png" width="500" />
</p>

<h2> <b> Example Neural Network Classifier Performance </b></h2>
We trained a Pytorch based Neural Network (multi-layer perceptron) to parse Suite2P suggested regions of interest (ROIs) into real vs not-real signal. In addition to training, our model was enhanced through Optuna (bayesian optimization) hyper-parameter tuning. 

First, traces are read into python and normalized to a pre-determined number of points.
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/exampletrace.png" width="500" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/examplenormtrace.png" width="500" /> 
</p>

Overall, the classifier performed moderatly well with a test F1 score >0.82:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/MLP_resultsds.png" width="500" />
</p>

The classifier was optimized via Bayesian Optimization available with the Optuna library:
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/newplot.png" width="500" />
</p>

Now, we quickly use this classifier to parse real from not-real ROIs suggested by Suite2P. Notably, not-real ROIs include ROIs containing no-signal or extremly low signal with S:N < 3:1. 
<p float="left">
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/RealSignal.png" width="500" />
  <img src="https://github.com/DJESTRIN/Sweet2Plus/blob/main/docs/images/Noise.png" width="500" /> 
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


