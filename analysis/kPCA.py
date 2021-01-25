#!/usr/bin/env python3

import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *
from sklearn.decomposition import KernelPCA


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate kernel PCA embedding.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--kernel', type=str, default='default',
        help='graph or preCalc.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:reaction_graph:targets\n'
        'examples: inchi:::tt\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional vector features with RBF kernel.\n' 
             'examples:\n'
             'red_T:0.1\n'
             'T,P:100,500'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    args = parser.parse_args()

    df_out = pd.read_csv(args.input, sep='\s+')
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    kernel_config = set_kernel_config(
        args.kernel, add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    params = {
        'train_size': None,
        'train_ratio': 1.0,
        'seed': 0,
    }
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        args.result_dir, args.input, kernel_config, properties, params
    )
    transformer = KernelPCA(n_components=2, kernel=kernel_config.kernel)
    embed = transformer.fit_transform(train_X)
    df_out['embed_X'] = embed[:, 0]
    df_out['embed_Y'] = embed[:, 1]
    df_out.to_csv('%s_embed_kPCA.log' % properties[0], sep=' ', index=False)


if __name__ == '__main__':
    main()
