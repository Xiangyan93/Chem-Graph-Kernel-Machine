# ChemML Executable Files
This directory contains the executable files. The input file should be formatted
 as [datasets](https://github.com/Xiangyan93/ChemML/tree/main/run/datasets). 

Both SMILES or InChI string are valid input of molecules.

Labeled reaction smarts string are valid input of chemical reactions.

Note that the posterior uncertainty scale from 0 to 1 when using normalized 
kernel.

## Single-Valued Property
1. The file datasets/critical-sim.txt contains the critical temperature and
critical density of molecules obtained from molecular dynamics (MD) simulation.
It is suggested to use a preCalc kernel as show in step 2-4. Use graph kernel
directly will be 2-3 times slower due to repeated calculations, as shown in 
step 5.

2. Preparation
    - Transfer the SMILES or inchi into graph object.
        ```
        python3 txt2pkl.py --result_dir tc -i datasets/critical-sim.txt --input_config SMILES:::tc --ntasks 6
        ```
3. Kernel Calculation
    - For fixed hyperparameters, it is fast to calculate the kernel matrix first.
        ```
        python3 KernelCalc.py --result_dir tc --input_config SMILES_sg:::tc --json_hyper ../hyperparameters/TMG.json
        ```
4. Performance evaluation
    - The training set ratio is 0.8. test-0.log are output. And
        You can set 1.0 to build a model using all data.
        ```
        python3 GPR.py --result_dir tc --gpr graphdot:none --kernel preCalc:0.01 --input_config SMILES_sg:::tc --train_test_config train_test::0.8:0
        ```
5. Hyperparameters optimization
    - Step 2-4 can be done by 1 command, but it is 2-3 times slower since the 
        graph kernel are not saved and repeatly computed.
        ```
        python3 GPR.py --result_dir tc --gpr graphdot:none --kernel graph:0.01 --input_config SMILES_sg:::tc --train_test_config train_test::0.8:0 -i datasets/critical-sim.txt --json_hyper ../hyperparameters/TMG.json
        ```
    - Hyperparameters optimization is not suggested since it is frequently to 
        get weired hyperparameters. Two types of optimization are available:
        1. sklearn, maximize the log marginal likelihood
            ```
            python3 GPR.py --result_dir tc --gpr sklearn:fmin_l_bfgs_b --kernel graph:0.01 --input_config SMILES_sg:::tc --train_test_config train_test::0.1:0 -i datasets/critical-sim.txt --json_hyper ../hyperparameters/TMG.json
            ```
        2. graphdot, minimize the Leave-One-Out loss
            ```
            python3 GPR.py --result_dir tc --gpr graphdot:L-BFGS-B --kernel graph:0.01 --input_config SMILES_sg:::tc --train_test_config train_test::0.1:0 -i datasets/critical-sim.txt --json_hyper ../hyperparameters/TMG.json
            ```
6. Prediction
    - Convert the model.pkl from preCalc kernel to graph kernel. This step is 
        only needed when you prepare the model through step 2-4.
        ```
        python3 preCalc2graph.py --result_dir tc --gpr graphdot:none --input_config SMILES_sg:::tc --json_hyper ../hyperparameters/TMG.json
        ```
    - Prepare a file of molecules to be predicted formatted as datasets/predict.txt.
        the results are save in predict.log.
        ```
        python3 predict.py --result_dir tc --gpr graphdot:none -i datasets/predict.txt --input_config SMILES_sg::: --json_hyper ../hyperparameters/TMG.json --f_model tc/model.pkl
        ```

## Temperature-Dependent Property
1. The file datasets/slab-sim.txt contains the VLE density and surface tension of 
molecular liquids obtained from molecular dynamics (MD) simulation. It is 
dependent on temperature.

2. Preparation
    - Transfer the SMILES or inchi into graph object.
        ```
        python3 txt2pkl.py --result_dir st -i datasets/slab-sim.txt --input_config SMILES:::st --ntasks 6
        ```
3. Kernel Calculation
    - For fixed hyperparameters, it is fast to calculate the kernel matrix first.
        ```
        python3 KernelCalc.py --result_dir st --input_config SMILES_sg:::st --json_hyper ../hyperparameters/TMG.json
        ```
4. Performance evaluation
    - The training set ratio is 0.8. test-0.log are output. And
        You can set 1.0 to build a model using all data.
        ```
        python3 GPR.py --result_dir st --gpr graphdot:none --kernel preCalc:0.01 --input_config SMILES_sg:::st --train_test_config train_test::0.8.:0 --add_features T:100
        ```
5. Hyperparameters optimization
    - The correctness of hyperparameters optimization in this case is not verified.
6. Prediction
    - Convert the model from preCalc kernel to graph kernel.
        ```
        python3 preCalc2graph.py --result_dir st --gpr graphdot:none --input_config SMILES_sg:::st --json_hyper ../hyperparameters/TMG.json --add_features T:100
        ```
    - Prepare a file of molecules to be predicted formatted as datasets/predict_T.txt.
        the results are save in predict.log.
        ```
        python3 predict.py --result_dir st --gpr graphdot:none -i datasets/predict_T.txt --input_config SMILES_sg::: --json_hyper ../hyperparameters/TMG.json --f_model st/model.pkl --add_features T:100
        ```

7. Low Rank approximation
    - Use Nystrom low rank approximation
        ```
        python3 GPR.py --result_dir st --gpr graphdot_nystrom:none --kernel preCalc:0.01 --input_config SMILES_sg:::st --train_test_config train_test::0.8:0 --add_features T:100 --nystrom_config 1000
        python3 preCalc2graph.py --result_dir st --gpr graphdot_nystrom:none --input_config SMILES_sg:::st --json_hyper ../hyperparameters/TMG.json --add_features T:100
        python3 predict.py --result_dir st --gpr graphdot_nystrom:none -i datasets/predict_T.txt --input_config SMILES_sg::: --json_hyper ../hyperparameters/TMG.json --f_model st/model.pkl --add_features T:100
        ```
8. Consensus model
    - Use consensus model
        ```
        python3 GPR.py --result_dir st --gpr graphdot:none --kernel preCalc:0.01 --input_config SMILES_sg:::st --train_test_config train_test::0.8:0 --add_features T:100 --consensus_config 10:1000:10:weight_uncertainty
        python3 preCalc2graph.py --result_dir st --gpr graphdot:none --input_config SMILES_sg:::st --json_hyper ../hyperparameters/TMG.json --add_features T:100 --consensus_config 10:1000:10:weight_uncertainty
        python3 predict.py --result_dir st --gpr graphdot:none -i datasets/predict_T.txt --input_config SMILES_sg::: --json_hyper ../hyperparameters/TMG.json --f_model st/model.pkl --add_features T:100 --consensus_config 10:1000:1:weight_uncertainty
        ```
9. Active Learning
    - Use supervised active learning as example. Unsupervised active learning 
        is also supported. 
        
        Do not optimize the hyperparameters and use preCalc 
        kernel during the active learning process! 
        ```
        python3 GPR_active.py --result_dir st --gpr graphdot:none --kernel preCalc:0.01 --input_config SMILES_sg:::st --train_test_config train_test::0.8:0 --add_features T:100 --active_config supervised:nlargest:5:1:500:0:200:100
        python3 preCalc2graph.py --result_dir st --gpr graphdot:none --input_config SMILES_sg:::st --json_hyper ../hyperparameters/TMG.json --add_features T:100
        python3 predict.py --result_dir st --gpr graphdot:none -i datasets/predict_T.txt --input_config SMILES_sg::: --json_hyper ../hyperparameters/TMG.json --f_model st/model.pkl --add_features T:100
        ```
    - You can extend the active learning process as following:
        ```
        python3 GPR_active.py --result_dir st --gpr graphdot:none --kernel preCalc:0.01 --input_config SMILES_sg:::st --train_test_config train_test::0.8:0 --add_features T:100 --active_config supervised:nlargest:5:1:1000:0:200:100 --continued
        ```

## Chemical Reaction Classification
1. The file datasets/reaction_classification.txt contains 3000 reactions of 
three types.

2. Preparation
    - Transfer the reaction_smarts into graph object.
        ```
        python3 txt2pkl.py --result_dir rxn -i datasets/reaction_classification.txt --input_config ::reaction_smarts:reaction_type --n_jobs 6
        ```

3. Kernel Calculation
    - For fixed hyperparameters, it is fast to calculate the kernel matrix first.
        ```
        python3 KernelCalc.py --result_dir rxn --input_config reaction_smarts_sg:::reaction_type --json_hyper ../hyperparameters/reaction.json
        ```
4. Performance evaluation
    - Using Gaussian process classification.
        ```
        python3 GPC.py --result_dir rxn --gpc sklearn:none --kernel preCalc -i datasets/reaction_classification.txt --input_config reaction_smarts_sg:::reaction_type --train_test_config train_test::0.2:0 -n 6
        ```
    - Using Support vector machine classification.
        ```
        python3 SVC.py --result_dir rxn --svc sklearn --kernel preCalc:1 -i datasets/reaction_classification.txt --input_config reaction_smarts_sg:::reaction_type --train_test_config train_test::0.2:0
        ```
      
      
# Kernel Computation in Blocks
1. For large data sets, it is convenient to calculate the kernel matrix in blocks 
and then concatenate them. A example is given for chemical reaction kernel.
The following commands are equivalent to the Step 3 in Chemical Reaction Classification.

2. Compute a sub-block of graph kernel matrix
    - Kernel matrix of reaction
        ```
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg:::reaction_type --block_config 1500:0,0 --json_hyper ../hyperparameters/reaction.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg:::reaction_type --block_config 1500:0,1 --json_hyper ../hyperparameters/reaction.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg:::reaction_type --block_config 1500:1,1 --json_hyper ../hyperparameters/reaction.json
        ```
      
    - Kernel matrix of reagents
        ```
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,0 --json_hyper ../hyperparameters/TMG.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,1 --json_hyper ../hyperparameters/TMG.json
        python3 ComputeGraphKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_agents_sg:::reaction_type --block_config 1500:1,1 --json_hyper ../hyperparameters/TMG.json
        ```
      
3. Compute a sub-block of hybrid kernel matrix
    - Hybrid kernel matrix of reaction and reagents
        ```
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,0
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:0,1
        python3 ComputeKernelBlock.py --result_dir rxn -i datasets/reaction_classification.txt --input_config reaction_smarts_sg,reaction_smarts_agents_sg:::reaction_type --block_config 1500:1,1
        ```

4. Compute the final kernel matrix
    - Concatenate all the blocks of kernel matrix
        ```
        python3 ConcatBlockKernels.py --result_dir rxn --block_config 1500:2,2
        ```