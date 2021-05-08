# Chem-Graph-Kernel-Machine
Predicting molecular properties using Marginalized Graph Kernel, [GraphDot](https://github.com/yhtang/GraphDot).

It supports regression (GPR) and classification (GPC, SVM) tasks on
* pure compounds.
* mixtures.
* chemical reactions.

Besides molecular graph, additional vector could also be added as input, such as 
temperature, pressure, etc.
## Installation
GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda env create -f environment.yml
```
## Usages
1. The executable files are in directory [run](https://github.com/Xiangyan93/ChemML/tree/main/run).
2. Notebook examples are in directory [notebook](https://github.com/Xiangyan93/ChemML/tree/main/notebook).
3. The directory [analysis](https://github.com/Xiangyan93/ChemML/tree/main/analysis) provide t-SNE and kPCA analysis using the nMGK.
4. The hyperparameter files in json format are placed in directory [hyperparameters](https://github.com/Xiangyan93/ChemML/tree/main/hyperparameters)
