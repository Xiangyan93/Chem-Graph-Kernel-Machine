# ChemML
Predicting thermodynamic properties of pure chemical compounds using Gaussian
Process Regressor-Marginalized Graph Kernel (GPR-MGK).  


## Requirements
Firstly, you need to install GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda install -c rdkit rdkit=2020.03.3.0
pip install scikit-learn==0.23.0
pip install graphdot==0.7
```
## Usages
1. Edit config.py to set up the nodes and edges features in graph.
  
2. Single-valued proeprty: triple point temperature.  
    - regression with fixed hyper-parameters.
        ```
        python3 run/GPR.py --gpr graphdot --optimizer None --kernel graph -i run/examples/tt.txt --property tt --result_dir tt-regression --alpha 0.01 --mode loocv --normalized
        ```
    - Optimize hyper-parameters, and regression.
        ```
        python3 run/GPR.py --gpr graphdot --optimizer L-BFGS-B --kernel graph -i run/examples/tt.txt --property tt --result_dir tt-regression --alpha 0.01 --mode loocv --normalized
        ```
    - Active learning
        ```
        python3 run/GPR_active.py --gpr graphdot --optimizer None --kernel graph -i run/examples/tt.txt --property tt --result_dir tt-active --alpha 0.01 --normalized --train_size 900 --learning_mode supervised --add_mode nlargest --init_size 5 --add_size 1 --max_size 200 --stride 100
        ```
        you can also extend the active learning
        ```
        python3 run/GPR_active.py --gpr graphdot --optimizer None --kernel graph -i run/examples/tt.txt --property tt --result_dir tt-active --alpha 0.01 --normalized --train_size 900 --max_size 500 --continued
        ```
    - Prediction of unknown molecule using existed model
       ```
       python3 run/predict.py --gpr graphdot --smiles CCCCCCCCCC --f_model run/tt-regression/model.pkl --normalized
       ```
    - Use Morgan fingerprints GPR-MF.
       ```
       python3 run/GPR.py --gpr graphdot --optimizer L-BFGS-B --kernel vector -i run/examples/tt.txt --property tt --result_dir tt-mf128b --alpha 0.01 --mode loocv --vectorFPparams morgan,2,128,0
       ```
       GPR-MF is inaccurate and computational expensive for hyperparameter 
       optimization.
   
3. Temperature, pressure-dependent properties.
    - Active learning
       ```
       python3 run/GPR_active.py --gpr graphdot --optimizer None --kernel graph -i run/examples/density.txt --add_features T,P --add_hyperparameters 100,500 --property density --result_dir density-active --alpha 0.01 --normalized --train_ratio 1.0 --learning_mode supervised --add_mode nlargest --init_size 5 --add_size 1 --max_size 200 --stride 100
       ```