#!/usr/bin/env python3
"""Plot Wall time (mean +/- stdev) over a push timeline from plot.csv."""

import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    labels, means, stdevs = [], [], []
    with open(args.input) as f:
        for row in csv.DictReader(f):
            labels.append(row['push'])
            means.append(float(row['mean_us']) / 1000.0)
            stdevs.append(float(row['stdev_us']) / 1000.0)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    ax.errorbar(range(len(labels)), means, yerr=stdevs, fmt='o-', capsize=4, color='tab:blue')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Push')
    ax.set_ylabel('Wall time (ms)')
    ax.set_title(f'Wall time (mean ± stdev) over last {len(labels)} pushes')
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)


if __name__ == '__main__':
    main()
