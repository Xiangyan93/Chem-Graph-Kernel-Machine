#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.kernels.GraphKernel import GraphKernelConfig
from chemml.graph.hashgraph import HashGraph


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
    )
    parser.add_argument(
        '--smiles', type=str,
        help='',
    )
    parser.add_argument(
        '--f_model', type=str,
        help='model.pkl',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    args = parser.parse_args()
    if args.gpr == 'graphdot':
        from chemml.GPRgraphdot.gpr import GPR
        GaussianProcessRegressor = GPR
    elif args.gpr == 'sklearn':
        from chemml.GPRsklearn.gpr import RobustFitGaussianProcessRegressor
        GaussianProcessRegressor = RobustFitGaussianProcessRegressor
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % args.gpr)
    kernel_config = GraphKernelConfig(
        NORMALIZED=args.normalized,
    )
    model = GaussianProcessRegressor.load_cls(args.f_model,
                                              kernel_config.kernel)
    X = [HashGraph.from_smiles(args.smiles)]
    y, y_std = model.predict(X, return_std=True)
    print('value: ', y[0], ';uncertainty: ', y_std[0])


if __name__ == '__main__':
    main()
