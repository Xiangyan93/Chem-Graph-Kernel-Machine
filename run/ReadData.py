#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml import CommonArgs
from chemml.data import Dataset


def main(args: CommonArgs) -> None:
    print('Preprocessing Dataset.')
    if args.data_path is not None:
        dataset = Dataset.from_csv(args)
    else:
        assert args.data_public is not None
        dataset = Dataset.from_public(args)
    dataset.save(args.save_dir, overwrite=True)
    print('Preprocessing Dataset Finished.')


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
