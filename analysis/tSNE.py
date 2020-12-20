#!/usr/bin/env python3

import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *
from sklearn.manifold import TSNE


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate tSNE embedding.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--kernel', type=str, default='default',
        help='The kernel.pkl file.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    args = parser.parse_args()

    kernel = pickle.load(open(args.kernel, 'rb'))
    R = kernel['K']
    d = R.diagonal() ** -0.5
    K = d[:, None] * R * d[None, :]
    D = np.sqrt(np.maximum(0, 2-2 * K**0.1))
    embed = TSNE(n_components=2).fit_transform(D)
    df = pd.read_csv(args.input, sep='\s+')
    df['embed_X'] = embed[:, 0]
    df['embed_Y'] = embed[:, 1]
    df['group_id'] = kernel['group_id']
    df.to_csv('embed.log', sep=' ', index=False)


if __name__ == '__main__':
    main()
