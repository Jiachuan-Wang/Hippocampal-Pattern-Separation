Code for the manuscript "Cerebellin-4 suppresses activity-dependent hippocampal neurogenesis to promote pattern separation memory".

## System requirements
### Hardware Requirements
The code requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 8 GB of RAM. 
The runtimes below are generated using a computer with 15 GB RAM, 24 cores@2.20 GHz.

### OS Requirements
The code has been tested on the following systems:
- Linux: CentOS Linux 7
- Windows: Windows 11
### Python Dependencies
I used Python 3.9.12.
```
numpy==1.22.4
pandas==1.4.4
matplotlib==3.5.2
scikit-learn==1.1.2
multiprocessing==2.6.2.1
```
### R Dependencies
I used R version 4.3.1.
```
abind_1.4-5
latex2exp_0.4.0
ggplot2_2.2.1
irlba_2.3.1
Matrix_1.2-3
MASS_7.3-47
randomForest_4.6-12
```
## Installation guide
```
git clone --recursive https://github.com/Jiachuan-Wang/Hippocampal-Pattern-Separation.git
```
The code should take approximately ?1 seconds to install.

## Instructions for use
Run the Python scripts to simulate the experiments. The generated data sets can be analyzed and visualized using the R codes.
To tweak the model parameters, you can specify them in `hp` object.
- `model.py` The main model components.
- `PS-only.py` Simulation of the Pattern Separation (PS) training task, related to Figure 3b, 3g, 3h.
- `FC+PS.py` Simulation of the full fear conditioning (FC) + PS task with freezing behavior, related to Figure 3j-l.
- `PCA 2cx.py` Plot the principle components (PCs) during a 2 contexts pattern separation task, related to Figure 3e.
- `PCA 4cx.py` Plot the PCs during a 4 contexts pattern separation task, related to Supplementary figure 3.
- `PCA 8cx.py` Plot the PCs during a 8 contexts pattern separation task, related to Supplementary figure 4.
- `Scree plot.py` Simulation of explain variance in dentate gyrus (DG) layer activity, related to Figure 3d and Supplementary figure 5 and 6.
- `Sparsity.py` Simulation of the active fraction of DG neurons, related to Figure 3f, 3i and Supplementary figure 7.
- `visualization.R` Use the simulated data sets to plot the figures.

The exact data sets used in the manuscript can be found in data availablity section and use the visualization code to generate the same figures. Run the code with different seeds can generate qualitatively similar figures. 

## Demo
Run `FC+PS.py` to generate the main results regarding the learning dynamics of cosine similarity and freezing ratios given different learning rates and neurogenesis situations. This would output a series of `csv` files starting with CosSimRes or FreezingRes indicting cosine similarity and freezing ratios on each training day, respectively. The code includes 500 replications for each of the 27 parameter sets, which takes approximately 4 hrs on a recommended computer. A sample data set is given in the `demo` folder, which can be used for plotting in `visualization.R`. 
