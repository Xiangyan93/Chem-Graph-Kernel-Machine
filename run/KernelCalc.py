#!/usr/bin/env python3
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import *
from chemml.kernels.KernelConfig import get_Xgroupid_from_df



def main():
    parser = argparse.ArgumentParser(
        description='Calculate Kernel Matrix and Gradients.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:targets\n'
        'examples: inchi::tt\n'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    args = parser.parse_args()

    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    # set kernel_config
    kernel_config = set_kernel_config(
        'graph', None, None,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    # set kernel_config
    df = get_df(args.input,
                os.path.join(args.result_dir, '%s.pkl' % ','.join(properties)),
                single_graph, multi_graph, reaction_graph)
    X, group_id = get_Xgroupid_from_df(df, kernel_config)
    print('**\tCalculating kernel matrix\t**')
    kernel_dict = {
        'group_id': group_id,
        'K': kernel_config.kernel(X),
        'theta': kernel_config.kernel.theta
    }
    print('**\tEnd Calculating kernel matrix\t**')
    kernel_pkl = os.path.join(args.result_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'))


if __name__ == '__main__':
    main()
