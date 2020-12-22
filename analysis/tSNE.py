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
        '-i', '--input', type=str, help='Input data in csv,pkl format.'
    )
    args = parser.parse_args()

    kernel = pickle.load(open(args.kernel, 'rb'))
    f1, f2 = args.input.split(',')
    df = pd.read_csv(f1, sep='\s+')
    df_ = pd.read_pickle(f2)
    df['id'] = df_['id']
    df['group_id'] = df_['group_id']
    gid = df_['group_id']
    R = kernel['K'][gid][:, gid]
    d = R.diagonal() ** -0.5
    K = d[:, None] * R * d[None, :]
    D = np.sqrt(np.maximum(0, 2-2 * K**2))
    embed = TSNE(n_components=2).fit_transform(D)
    df['embed_X'] = embed[:, 0]
    df['embed_Y'] = embed[:, 1]
    df.to_csv('embed.log', sep=' ', index=False)


if __name__ == '__main__':
    main()
