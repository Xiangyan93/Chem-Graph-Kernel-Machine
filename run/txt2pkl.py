#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *


def main():
    parser = argparse.ArgumentParser(
        description='Transform input file into pickle file, in which the InChI '
                    'or SMILES string was transformed into graphs.',
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
                                         'format: single_graph:multi_graph:reaction_graph:targets\n'
                                         'examples: inchi:::tt\n'
    )
    parser.add_argument(
        '-n', '--ntasks', type=int, default=cpu_count(),
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()

    # set result directory
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)

    # set kernel_config
    get_df(args.input,
           os.path.join(result_dir, '%s.pkl' % ','.join(properties)),
           single_graph, multi_graph, reaction_graph, n_process=args.ntasks)


if __name__ == '__main__':
    main()
