# Chem-Graph-Kernel-Machine
Predicting molecular properties of pure chemical compounds using Gaussian
Process Regression-normalized Marginalized Graph Kernel (GPR-nMGK).  

## Dependencies
GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda install -c rdkit rdkit=2020.03.3.0
pip install scikit-learn==0.23.0
pip install graphdot==0.7
pip install tqdm==4.50.0
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

