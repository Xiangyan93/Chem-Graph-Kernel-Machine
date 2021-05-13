# Chem-Graph-Kernel-Machine Executable Files
This directory contains the all executable files. 


## Data sets.
The input file should be formatted as [datasets](https://github.com/Xiangyan93/ChemML/tree/main/run/datasets). 

Both SMILES or InChI string are valid input of molecules.
Labeled reaction smarts string are valid input of chemical reactions.

## Marginalized Graph Kernel (MGK) Architecture.
The architecture of MGK and associated hyperparameters are defined in a file in 
JSON format. Several choices are provided in ../hyperparameters. "tMGR" use a
tensor-product architecture. "additive" use an additive architecture.
- "tMGR.json" is MGK with molecular-size-dependent normalization.
- "tMGR-Norm.json" is MGK with simple normalization.
- "tMGR-non-Norm.json" is MGK without normalization.

## Gaussian Process Regression (GPR) for Single-Valued Property
The file datasets/Public/freesolv.csv is the hydration free energy in water. We use
this data set as example.

1. Read Dataset.
    - Read the dataset and preprocess. Save result in freesolv/dataset.pkl.
        ```
        python3 ReadData.py --save_dir freesolv --data_path datasets/Public/freesolv.csv --pure_columns smiles --target_columns freesolv --n_jobs 6
        ```
2. Kernel Calculation
    - Calculate the entire kernel matrix, and saved. Use the non-optimized hyperparameters.
        ```
        python3 KernelCalc.py --save_dir freesolv --graph_kernel_type graph --graph_hyperparameters ../hyperparameters/tMGR.json
        ```
3. Performance evaluation
    - The training set ratio is 0.8. test-*.log are output. 
        ```
        python3 ModelEvaluate.py --save_dir freesolv --graph_kernel_type preCalc --dataset_type regression --model_type gpr --split_type random --split_sizes 0.8 0.2 --alpha 0.01 --metric rmse --extra_metrics r2 --num_folds 10
        ```
The performance is not good, use RDKit features and optimized hyperparameters for best performance.

## Gaussian Process Regression (GPR) for Temperature-Dependent Property
The file datasets/ThermoSIM/slab-sim.txt is the surface tension at different temperatures. 
We use this data set as example.
```
python3 ReadData.py --save_dir st --data_path datasets/ThermoSIM/slab-sim.txt --pure_columns smiles --target_columns st --feature_columns T --group_reading --n_jobs 6
python3 KernelCalc.py --save_dir st --graph_kernel_type graph --graph_hyperparameters ../hyperparameters/tMGR.json
python3 ModelEvaluate.py --save_dir st --graph_kernel_type preCalc --dataset_type regression --model_type gpr --split_type random --split_sizes 0.2 0.8 --alpha 0.01 --metric rmse --extra_metrics r2 --num_folds 10 --features_hyperparameters 100.0
```

## Use RDKit features
Use both molecular graph (MGK) and 200 molecular descriptors (RBF kernels) that calculated by RDKit as input.

The optimized hyperparameters are provided in datasets/Public/freesolv.
```
python3 ReadData.py --save_dir freesolv --data_path datasets/Public/freesolv.csv --pure_columns smiles --target_columns freesolv --n_jobs 6 --features_generator rdkit_2d_normalized
python3 KernelCalc.py --save_dir freesolv --graph_kernel_type graph --graph_hyperparameters datasets/Public/freesolv/hyperparameters_0.json --features_hyperparameters_file datasets/Public/freesolv/sigma_RBF.json
python3 ModelEvaluate.py --save_dir freesolv --graph_kernel_type preCalc --dataset_type regression --model_type gpr --split_type random --split_sizes 0.8 0.2 --alpha datasets/Public/freesolv/alpha --metric rmse --extra_metrics r2 --num_folds 10
```

## Classification
We use bbbp data set as an example.

Read data and calculate kernels.
```
python3 ReadData.py --save_dir bbbp --data_path datasets/Public/bbbp.csv --pure_columns smiles --target_columns p_np --n_jobs 6 --features_generator rdkit_2d_normalized
python3 KernelCalc.py --save_dir bbbp --graph_kernel_type graph --graph_hyperparameters datasets/Public/bbbp/hyperparameters_0.json --features_hyperparameters_file datasets/Public/bbbp/sigma_RBF.json
```
1. Gaussian Process Classification
```
python3 ModelEvaluate.py --save_dir bbbp --graph_kernel_type preCalc --dataset_type classification --model_type gpc --split_type random --split_sizes 0.8 0.2 --metric roc-auc --num_folds 10
```
2. Support Vector Machine Classification
```
python3 ModelEvaluate.py --save_dir bbbp --graph_kernel_type preCalc --dataset_type classification --model_type svc --split_type random --split_sizes 0.8 0.2 --C 1.0 --metric roc-auc --num_folds 10
```

## Classification for chemical reactions
SVC is faster than GPC, as well as lower memory costs.
```
python3 ReadData.py --save_dir rxn --data_path datasets/RxnClassification/test.csv --reaction_columns good_smarts --target_columns reaction_type --n_jobs 6
python3 KernelCalc.py --save_dir rxn --graph_kernel_type graph --graph_hyperparameters ../hyperparameters/reaction.json
python3 ModelEvaluate.py --save_dir rxn --graph_kernel_type preCalc --dataset_type multiclass --model_type svc --split_type random --split_sizes 0.8 0.2 --C 1.0 --metric accuracy --no_proba --num_folds 10
python3 ModelEvaluate.py --save_dir rxn --graph_kernel_type preCalc --dataset_type multiclass --model_type gpc --split_type random --split_sizes 0.8 0.2 --metric accuracy --no_proba --num_folds 10
```

## Hyperparameters Optimization
For regression tasks, it is suggested to minimize the LOOCV loss.

For classification tasks, it is suggested to minimize the roc-auc of 10-fold training/test data splits.

Best hyperparameters can be obtained by applying (1) multiple Bayesian optimization (global optimization) 
from different random seed, and then (2) Scipy optimization (local optimization).

1. Bayesian Optimization. Using hyperopt python package.
    - GPR without RDKit features.
        ```
        python3 HyperOpt.py --save_dir freesolv --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/additive.json --num_iters 100 --seed 0 --alpha 0.01 --alpha_bounds 0.008 0.02
        ```
    - GPR with RDKit features.
        ```
        python3 HyperOpt.py --save_dir freesolv --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/additive.json --num_iters 100 --seed 0 --alpha 0.01 --alpha_bounds 0.008 0.02 --features_hyperparameters 1.0 --features_hyperparameters_min 0.1 --features_hyperparameters_max 20.0
        ```
    - GPC with RDKit features.
        ```
        python3 HyperOpt.py --save_dir freesolv --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/additive.json --num_iters 100 --seed 0 --alpha 0.01 --alpha_bounds 0.008 0.02 --features_hyperparameters 1.0 --features_hyperparameters_min 0.1 --features_hyperparameters_max 20.0
        ```
2. Scipy Optimization.
   
   This is allowed only for regression tasks. 
   
   The data noise "alpha" in GPR is fixed.
   - GPR without RDKit features.
        ```
        python3 HyperOpt.py --save_dir freesolv --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/tMGR.json --seed 0 --alpha 0.01 --optimizer SLSQP
        ```
   - GPR with RDKit features.
        ```
        python3 HyperOpt.py --save_dir freesolv --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/tMGR.json --seed 0 --alpha 0.01 --optimizer SLSQP --features_hyperparameters 1.0 --features_hyperparameters_min 0.1 --features_hyperparameters_max 20.0
        ```
## Active Learning
1. Supervised active learning.
    ```
    python3 ActiveLearning.py --save_dir freesolv --graph_kernel_type preCalc --dataset_type regression --model_type gpr --alpha datasets/Public/freesolv/alpha --metric rmse --extra_metrics r2 --learning_algorithm supervised --initial_size 2 --add_size 1 --stop_size 400 --evaluate_stride 50
    ```
2. Unsupervised active learning.
    ```
    python3 ActiveLearning.py --save_dir freesolv --graph_kernel_type preCalc --dataset_type regression --model_type gpr --alpha datasets/Public/freesolv/alpha --metric rmse --extra_metrics r2 --learning_algorithm unsupervised --initial_size 2 --add_size 1 --stop_size 400 --evaluate_stride 50
    ```
<div align="center">
<p><img src="../docs/picture/active_learning.png" width="1000"/></p>
</div> 


## Data Embedding.
1. tSNE.
```
python3 Embedding.py --save_dir freesolv --graph_kernel_type preCalc --embedding_algorithm tSNE --save_png --n_jobs 6
```
2. kPCA.
```
python3 Embedding.py --save_dir freesolv --graph_kernel_type preCalc --embedding_algorithm kPCA --save_png --n_jobs 6
```
## Kernel Computation in Blocks
1. For large data sets, it is convenient to calculate the kernel matrix in blocks 
and then concatenate them. A example is given for chemical reaction kernel.
The following commands are equivalent to the Step 3 in Chemical Reaction Classification.

2. Compute a sub-block of graph kernel matrix
    - Kernel matrix of reaction
        ```
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg:::reaction_type --block_config 1500:0,0 --json_hyper ../hyperparameters/reaction.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg:::reaction_type --block_config 1500:0,1 --json_hyper ../hyperparameters/reaction.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg:::reaction_type --block_config 1500:1,1 --json_hyper ../hyperparameters/reaction.json
        ```
      
    - Kernel matrix of reagents
        ```
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,0 --json_hyper ../hyperparameters/tMGR.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,1 --json_hyper ../hyperparameters/tMGR.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:1,1 --json_hyper ../hyperparameters/tMGR.json
        ```
      
3. Compute a sub-block of hybrid kernel matrix
    - Hybrid kernel matrix of reaction and reagents
        ```
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,0
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,1
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/RxnClassification/test_3000.csv --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:1,1
        ```

4. Compute the final kernel matrix
    - Concatenate all the blocks of kernel matrix
        ```
        python3 ConcatBlockKernels.py --result_dir rxn --block_config 1500:2,2
        ```