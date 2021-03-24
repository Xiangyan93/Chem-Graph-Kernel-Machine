#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import warnings
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml import CommonArgs
from chemml.data import Dataset


def main(args: CommonArgs) -> None:
    print('Preprocessing Dataset.')
    # set result directory
    result_dir = args.save_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    dataset = Dataset.from_csv(args)
    dataset.save(args.save_dir, overwrite=True)
    print('Preprocessing Dataset Finished.')


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
