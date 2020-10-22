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
1. Use hyperparameters/generator.py to set up the nodes and edges features in graph.
  
2. Single-valued proeprty: triple point temperature.  
    - regression with fixed hyper-parameters.
        ```
        python3 run/GPR.py --result_dir run/tt-regression --gpr graphdot:none --kernel graph:True:0.01 -i run/examples/tt.txt --input_config inchi::tt --train_test_config loocv:::0 --json_hyper hyperparameters/tensorproduct.json
        ```
    - Optimize hyper-parameters, and regression.
        ```
        python3 run/GPR.py --result_dir run/tt-regression --gpr graphdot:L-BFGS-B --kernel graph:True:0.01 -i run/examples/tt.txt --input_config inchi::tt --train_test_config loocv:::0 --json_hyper hyperparameters/tensorproduct.json
        ```
    - Active learning
        ```
        python3 run/GPR_active.py --result_dir run/tt-active --gpr sklearn:none --kernel graph:true:0.01 -i run/examples/tt.txt --input_config inchi::tt --train_test_config ::0.9: --active_config supervised:nlargest:5:1:200:0:200:100 --json_hyper hyperparameters/tensorproduct.json
        ```
        you can also extend the active learning
        ```
        python3 run/GPR_active.py --result_dir run/tt-active --gpr sklearn:none --kernel graph:true:0.01 -i run/examples/tt.txt --input_config inchi::tt --train_test_config ::0.9: --active_config supervised:nlargest:5:1:500:0:200:100 --json_hyper hyperparameters/tensorproduct.json --continued
        ```
    - Prediction of unknown molecule using existed model
       ```
       python3 run/predict.py --gpr graphdot --normalized -i run/examples/tt_predict.txt --input_config SMILES:: --json_hyper run/tt-regression/hyperparameters.json --f_model run/tt-regression/model.pkl
       ```
       Put the molecules to be predicted in a file as run/examples/tt_predict.txt, 
       "predict.csv" will be generated for the prediction results.
   
3. Temperature, pressure-dependent properties.
    - Active learning
       ```
       python3 run/GPR_active.py --result_dir run/density-active --gpr sklearn:none --kernel graph:true:0.01 -i run/examples/density.txt --input_config SMILES::density --add_features T,P:100,500 --train_test_config ::1.0: --active_config supervised:nlargest:5:1:200:0:200:50 --json_hyper hyperparameters/tensorproduct.json
       ```
      
4. Separate CPU and GPU calculation with fixed hyper-parameters.  
    - GPR
        ```
        CPU: python3 run/txt2pkl.py --result_dir run/tt-regression -i run/examples/tt.txt --input_config inchi::tt
        GPU: python3 run/KernelCalc.py --result_dir run/tt-regression -i run/examples/tt.txt --input_config inchi::tt --normalized --json_hyper hyperparameters/tensorproduct.json
        CPU: python3 run/GPR.py --result_dir run/tt-regression --gpr sklearn:none --kernel preCalc::0.01 -i run/examples/tt.txt --train_test_config loocv:::0 --json_hyper hyperparameters/tensorproduct.json
        GPU: python3 run/preCalc2graph.py --result_dir run/tt-regression --gpr sklearn --normalized --input_config inchi::tt --json_hyper hyperparameters/tensorproduct.json
        ```
    
    - Active Learning
        ```
        
        ```