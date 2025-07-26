Code for the manuscript "Cerebellin-4 suppresses activity-dependent hippocampal neurogenesis to promote pattern separation memory".

## System requirements
### Hardware Requirements
The code can be run on a standard computer with sufficient RAM to support the defined operations.
Benchmark runtimes below were generated using a server with 15 GB RAM and 24 cores @ 2.20 GHz.

### OS Requirements
The code has been tested on the following systems:
- Linux: CentOS Linux 7
- Windows: Windows 11
### Python Dependencies
Tested with Python 3.9.12.
Install the following packages:
```
numpy==1.22.4
pandas==1.4.4
matplotlib==3.5.2
scikit-learn==1.1.2
multiprocessing==2.6.2.1
```
### R Dependencies
Tested with R version 4.3.1.
Install the following packages:
```
tidyverse_2.0.0
ggplot2_3.5.1
scales_1.3.0
viridis_0.6.4
latex2exp_0.9.6
```
## Installation guide
Clone the repository:
```
git clone --recursive https://github.com/Jiachuan-Wang/Hippocampal-Pattern-Separation.git
```
Installation takes approximately 1 second.

## Instructions for use
Run the Python scripts to simulate the experiments. The generated datasets can be analyzed and visualized using the R script.
Model parameters can be modified in the `hp` object.

Script Overview:
- `model.py` Main model components
- `PS-only.py` Simulates the Pattern Separation (PS) training task (Figure 3b, 3g, 3h)
- `FC+PS.py` Simulates full Fear Conditioning (FC) + PS task with freezing behavior (Figure 3j–l)
- `PCA 2cx.py` Plots principal components (PCs) during a 2-context PS task (Figure 3e)
- `PCA 4cx.py` Plots PCs during a 4-context PS task (Supplementary Figure 3)
- `PCA 8cx.py` Plots PCs during an 8-context PS task (Supplementary Figure 4)
- `Scree plot.py` Simulates explained variance in dentate gyrus (DG) activity (Figure 3d; Supplementary Figures 5 and 6)
- `Sparsity.py` Simulates active fraction of DG neurons (Figure 3f, 3i; Supplementary Figure 7)
- `visualization.R` Generates figures from simulated datasets

The exact datasets used in the manuscript will be available to reviewers and readers (see Reporting Summary). Then you can use the provided `visualization.R` script to recreate the published figures. Running the simulations with different seeds yields qualitatively similar results.

## Demo
To reproduce the main results on learning dynamics (cosine similarity and freezing ratios across different learning rates and neurogenesis conditions), run:
```
python3 FC+PS.py
```
This script will output a series of `.csv` files with names starting with CosSimRes or FreezingRes, indicating cosine similarity and freezing ratios for each training day, respectively.

The code includes 500 replications for each of 27 parameter sets, and takes approximately 4 hours on a recommended machine.

A sample dataset is provided in the `demo` folder and can be used directly with `visualization.R` for plotting.
