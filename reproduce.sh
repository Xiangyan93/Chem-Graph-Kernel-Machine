# Preparation.
# (GCC 7.* or 9.*)(CUDA >= 10.1) are required for graphdot.
git clone -b comparative_study https://github.com/Xiangyan93/Chem-Graph-Kernel-Machine
git clone -b gpr-mgk-comparison https://github.com/Xiangyan93/chemprop
git clone -b uncertainty_quantification https://github.com/Xiangyan93/chemprop chemprop_uq
conda env create -f Chem-Graph-Kernel-Machine/environment.yml
conda activate graphdot
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install tensorboardX GPy seaborn

# Notice that running this script completely in one computer may take several weeks.

# regression data sets
properties=(delaney freesolv lipo pdbbind_core pdbbind_refined pdbbind_full qm7)
metrics=(rmse rmse rmse rmse rmse rmse mae)
for((i=0;i<7;++i))
do
  property=${properties[$i]}

  ### GP-MGK hyperparameter optimization.
  cd Chem-Graph-Kernel-Machine/run
  # python ReadData.py --save_dir $property --data_path datasets/Public/$property.csv --pure_columns smiles --n_jobs 6 --features_generator rdkit_2d_normalized
  # The hyperparameters are optimized using HyperOpt.py.
  python HyperOpt.py --save_dir $property --graph_kernel_type graph --dataset_type regression --model_type gpr --split_type loocv --metric rmse --num_folds 1 --graph_hyperparameters ../hyperparameters/additive.json --num_iters 100 --seed 0 --alpha 0.01 --alpha_bounds 0.008 0.02 --features_hyperparameters 1.0 --features_hyperparameters_min 0.1 --features_hyperparameters_max 20.0
  # hyperparameters_0.json, sigma_RBF.json, alpha are saved hyperparameter files, which will be generated in
  # directory $property.
  python KernelCalc.py --save_dir $property --graph_kernel_type graph --graph_hyperparameters datasets/Public/$property/hyperparameters_0.json --features_hyperparameters_file datasets/Public/$property/sigma_RBF.json
  cd ../../

  ### D-MPNN hyperparameter optimization.
  python chemprop/hyperparameter_optimization.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type regression --config_save_dir $property\_config --num_iters 20 --split_type random --split_size 0.8 0.1 0.1 --metric $metric --num_folds 10 --features_generator rdkit_2d_normalized --no_features_scaling --num_workers 6
  # Hyperparameter optimization is time-consuming, the optimized hyperparameters are saved in datasets/Public.

  ### Prediction Accuracy comparison, and Uncertainty Quantification of GPR-MGK
  metric=${metrics[$i]}
  # random split
  python chemprop/train.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type regression --save_dir chemprop/$property-random --split_type random --split_size 0.8 0.1 0.1 --metric $metric --extra_metric r2 --num_folds 100 --ensemble_size 5 --config_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property/$property\_config --epoch 50 --save_preds --features_generator rdkit_2d_normalized --no_features_scaling --gp --kernel Chem-Graph-Kernel-Machine/run/$property/kernel.pkl --num_workers 6 --alpha Chem-Graph-Kernel-Machine/run/datasets/Public/$property/alpha
  # scaffold split
  python chemprop/train.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type regression --save_dir chemprop/$property-scaffold --split_type scaffold --split_size 0.8 0.1 0.1 --metric $metric --extra_metric r2 --num_folds 100 --ensemble_size 5 --config_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property/$property\_config --epoch 50 --save_preds --features_generator rdkit_2d_normalized --no_features_scaling --gp --kernel Chem-Graph-Kernel-Machine/run/$property/kernel.pkl --num_workers 6 --alpha Chem-Graph-Kernel-Machine/run/datasets/Public/$property/alpha
done

# classification data sets
properties=(bace bbbp clintox sider)
for((i=0;i<4;++i))
do
  property=${properties[$i]}

  ### GP-MGK hyperparameter optimization.
  cd Chem-Graph-Kernel-Machine/run
  python ReadData.py --save_dir $property --data_path datasets/Public/$property.csv --pure_columns smiles --n_jobs 6 --features_generator rdkit_2d_normalized
  # The hyperparameters are optimized using HyperOpt.py.
  python HyperOpt.py --save_dir $property --graph_kernel_type graph --dataset_type classification --model_type gpc --split_type random --metric roc-auc --num_folds 10 --graph_hyperparameters ../hyperparameters/additive.json --num_iters 100 --seed 0 --features_hyperparameters 1.0 --features_hyperparameters_min 0.1 --features_hyperparameters_max 20.0
  # hyperparameters_0.json, sigma_RBF.json, are saved hyperparameter files, which will be generated in
  # directory $property. It is time-consuming, the optimized hyperparameters are saved in datasets/Public.
  python KernelCalc.py --save_dir $property --graph_kernel_type graph --graph_hyperparameters datasets/Public/$property/hyperparameters_0.json --features_hyperparameters_file datasets/Public/$property/sigma_RBF.json
  cd ../../

  ### D-MPNN hyperparameter optimization.
  python chemprop/hyperparameter_optimization.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type classification --config_save_dir $property\_config --num_iters 20 --split_type random --split_size 0.8 0.1 0.1 --metric auc --num_folds 10 --features_generator rdkit_2d_normalized --no_features_scaling --num_workers 6
  # Hyperparameter optimization is time-consuming, the optimized hyperparameters are saved in datasets/Public.

  ### Prediction Accuracy comparison.
  metric=${metrics[$i]}
  # random split
  python chemprop/train.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type classification --save_dir chemprop/$property-random --split_type random --split_size 0.8 0.1 0.1 --metric auc --num_folds 100 --ensemble_size 5 --config_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property/$property\_config --epoch 50 --save_preds --features_generator rdkit_2d_normalized --no_features_scaling --gp --kernel Chem-Graph-Kernel-Machine/run/$property/kernel.pkl --num_workers 6
  # scaffold split
  python chemprop/train.py --data_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property.csv --dataset_type classification --save_dir chemprop/$property-scaffold --split_type scaffold --split_size 0.8 0.1 0.1 --metric auc --num_folds 100 --ensemble_size 5 --config_path Chem-Graph-Kernel-Machine/run/datasets/Public/$property/$property\_config --epoch 50 --save_preds --features_generator rdkit_2d_normalized --no_features_scaling --gp --kernel Chem-Graph-Kernel-Machine/run/$property/kernel.pkl --num_workers 6
done

### Uncertainty Quantification of D-MPNN MVE
cd chemprop_uq
tar xvzf data.tar.gz
python uncertainty_evaluation/populate_build.py
bash uncertainty_evaluation/populate.sh