# ChemML
This code is developed for machine learning tasks in chemistry. So far,
only kernel methods are valid. The Marginalized Graph Kernel (MGK) is computed 
using [GraphDot](https://github.com/yhtang/GraphDot) package.

Now, it supports:
1. Regression (GPR) and classification tasks(GPC, SVM).
2. Input can be pure compounds, mixtures and chemical reactions. 
Additional input such as temperature and pressure are also valid.

## Requirements
GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda install -c rdkit rdkit=2020.03.3.0
pip install scikit-learn==0.23.0
pip install graphdot==0.7
pip install tqdm==4.50.0
```
Recent revision haven't been merged into GraphDot yet, please use the 
GraphDot
```
git clone -b feature/start_probability https://gitlab.com/XiangyanSJTU/graphdot.git
```
## Usages
1. Executable files are in directory [run](https://github.com/Xiangyan93/ChemML/tree/main/run).
2. Notebook examples are in directory [notebook](https://github.com/Xiangyan93/ChemML/tree/main/notebook).
3. t-SNE and kPCA can be executed in directory [analysis](https://github.com/Xiangyan93/ChemML/tree/main/analysis).
4. Hyperparameter file are in directory [hyperparameters](https://github.com/Xiangyan93/ChemML/tree/main/hyperparameters)

