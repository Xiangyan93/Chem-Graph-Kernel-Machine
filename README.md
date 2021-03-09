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
conda create -n graphdot_dev python=3.7 -y
conda install -c rdkit rdkit=2020.03.3.0
pip install -r requirements.txt
```
Some revisions haven't been merged into GraphDot yet, please use the graphdot:
```
git clone -b feature/start_probability https://gitlab.com/XiangyanSJTU/graphdot.git
```
## Usages
1. The executable files are in directory [run](https://github.com/Xiangyan93/ChemML/tree/main/run).
2. Notebook examples are in directory [notebook](https://github.com/Xiangyan93/ChemML/tree/main/notebook).
3. The directory [analysis](https://github.com/Xiangyan93/ChemML/tree/main/analysis) provide t-SNE and kPCA analysis using the nMGK.
4. The hyperparameter files in json format are placed in directory [hyperparameters](https://github.com/Xiangyan93/ChemML/tree/main/hyperparameters)
