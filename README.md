# ChemML
Predicting molecular properties of pure chemical compounds using Gaussian
Process Regressor-Marginalized Graph Kernel (GPR-MGK).  

## Requirements
GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda install -c rdkit rdkit=2020.03.3.0
pip install scikit-learn==0.23.0
pip install graphdot==0.7
pip install tqdm==4.50.0
```
Part of the revision haven't been merged into graphdot yet, please use the 
graphdot
```
git clone -b feature/start_probability https://gitlab.com/XiangyanSJTU/graphdot.git
```
## Usages
1. Executable files are in directory [run](https://github.com/Xiangyan93/ChemML/tree/main/run).
2. Notebook examples are in directory [notebook](https://github.com/Xiangyan93/ChemML/tree/main/notebook).
3. t-SNE and kPCA can be executed in directory [analysis](https://github.com/Xiangyan93/ChemML/tree/main/analysis).
4. Hyperparameter file are in directory [hyperparameters](https://github.com/Xiangyan93/ChemML/tree/main/hyperparameters)

