# python3 csv_to_md.py --runs 41 --baseline baseline/nvidia-l40s-artifact/summary_profiler_nvidia-l40s.csv --current artifacts/nvidia-l40s-artifact/summary_profiler_nvidia-l40s.csv --alpha 0.05

import argparse
import csv
import math
import tabulate as tab
from scipy.stats import t as t_dist

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs', type=int, required=True, help='Number of retained runs (post-discard), used as the sample size for the significance test')
parser.add_argument('-b', '--baseline', required=True, help='Baseline CSV file')
parser.add_argument('-c', '--current', required=True, help='Current CSV file')
parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default 0.05 -> 95% CI)')
args = parser.parse_args()

def get_dict(csv_filename):
  with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    return {name: (float(mean), float(stdev), int(count)) for name, mean, stdev, count in csv_reader}

dict_baseline = get_dict(args.baseline)
dict_current = get_dict(args.current)

def welch(x, sx, m, y, sy, n):
  """Welch's t-test for difference of means d = x - y.

  x, sx, m: mean, stdev, sample size of "current"
  y, sy, n: mean, stdev, sample size of "baseline"

  Returns d, se, t, nu, ci_half_width
  """
  d = x - y
  if m < 2 or n < 2:
    return d, 0.0, 0.0, 0.0, 0.0
  var_x = sx**2 / m
  var_y = sy**2 / n
  se = math.sqrt(var_x + var_y)
  if se == 0.0:
    return d, 0.0, 0.0, float(m + n - 2), 0.0
  t = d / se
  nu = (var_x + var_y)**2 / (var_x**2 / (m - 1) + var_y**2 / (n - 1))
  t_star = t_dist.ppf(1.0 - args.alpha / 2.0, nu)
  ci_half_width = t_star * se
  return d, se, t, nu, ci_half_width

def get_emoji(d, ci_half_width):
  ci_low = d - ci_half_width
  ci_high = d + ci_half_width
  if ci_low > 0:
    return ':red_circle:'
  elif ci_high < 0:
    return ':green_circle:'
  else:
    return ':white_circle:'

table = []
for name in dict_current.keys() | dict_baseline.keys():
  current = dict_current.get(name)
  baseline = dict_baseline.get(name)

  if current is not None:
    mean, stdev, count = current
    sort_key = mean
    current_row = [int(mean), f'{stdev:.2f}', count]
  else:
    sort_key = float('-inf')
    current_row = ['', '', '']

  if baseline is not None:
    baseline_mean, baseline_stdev, baseline_count = baseline
    baseline_row = [int(baseline_mean), f'{baseline_stdev:.2f}', baseline_count]
  else:
    baseline_row = ['', '', '']

  if current is not None and baseline is not None:
    d, se, t, nu, ci_half_width = welch(mean, stdev, args.runs, baseline_mean, baseline_stdev, args.runs)
    emoji = get_emoji(d, ci_half_width)
    ci_low = d - ci_half_width
    ci_high = d + ci_half_width
    comparison_row = [f'{d:.2f}', f'[{ci_low:.2f}, {ci_high:.2f}]', emoji]
  else:
    comparison_row = ['', '', '']

  table.append((sort_key, [name, *current_row, *baseline_row, *comparison_row]))

table.sort(key = lambda row: row[0], reverse=True)
table = [row for _, row in table]

confidence_pct = int(round((1.0 - args.alpha) * 100))
header = [
  'name',
  'mean (\u03BCs)',
  'stdev \u03C3',
  'count/run',
  'mean (base)',
  'stdev (base)',
  'count/run (base)',
  'diff \u0394',
  f'{confidence_pct}% CI (\u0394)',
  '',
]
print(tab.tabulate(table, header, tablefmt="github"))