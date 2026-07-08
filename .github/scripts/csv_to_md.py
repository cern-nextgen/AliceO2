# python3 csv_to_md.py --runs 41 --baseline baseline/nvidia-l40s-artifact/summary_profiler_nvidia-l40s.csv --current artifacts/nvidia-l40s-artifact/summary_profiler_nvidia-l40s.csv --alpha 0.05

import argparse
import csv
import math
import tabulate as tab
from scipy.stats import t as t_dist

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs', type=int, required=True, help='Number of runs')
parser.add_argument('-b', '--baseline', required=True, help='Baseline CSV file')
parser.add_argument('-c', '--current', required=True, help='Current CSV file')
parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default 0.05 -> 95% CI)')
args = parser.parse_args()

def get_2d_list(csv_filename):
  with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    return [[str(name), float(mean), float(stdev), int(count)] for name, mean, stdev, count in csv_reader]

table_baseline = get_2d_list(args.baseline)
table_current = get_2d_list(args.current)

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

  # Welch-Satterthwaite degrees of freedom
  nu = (var_x + var_y)**2 / (var_x**2 / (m - 1) + var_y**2 / (n - 1))

  t_star = t_dist.ppf(1.0 - args.alpha / 2.0, nu)
  ci_half_width = t_star * se

  return d, se, t, nu, ci_half_width

def get_emoji(d, ci_half_width):
  # CI on d = current - baseline. If the whole CI lies above zero,
  # current is significantly slower (regression) -> red.
  # If the whole CI lies below zero, current is significantly faster -> green.
  # Otherwise zero is inside the CI -> not significant -> white.
  ci_low = d - ci_half_width
  ci_high = d + ci_half_width
  if ci_low > 0:
    return ':red_circle:'
  elif ci_high < 0:
    return ':green_circle:'
  else:
    return ':white_circle:'

table = []
for baseline, current in zip(table_baseline, table_current):
  baseline_name, baseline_mean, baseline_stdev, count_baseline = baseline
  name, mean, stdev, count = current
  assert(baseline_name == name)
  total_time_baseline = baseline_mean * (count_baseline // args.runs)
  total_time = mean * (count // args.runs)
  d, se, t, nu, ci_half_width = welch(mean, stdev, count, baseline_mean, baseline_stdev, count_baseline)
  emoji = get_emoji(d, ci_half_width)
  ci_low = d - ci_half_width
  ci_high = d + ci_half_width
  table.append([
    name,
    int(total_time),
    int(mean),
    f'{stdev:.2f}',
    count,
    int(total_time_baseline),
    int(baseline_mean),
    f'{baseline_stdev:.2f}',
    count_baseline,
    f'{d:.2f}',
    f'[{ci_low:.2f}, {ci_high:.2f}]',
    emoji,
  ])

table.sort(key = lambda row: row[1], reverse=True)

confidence_pct = int(round((1.0 - args.alpha) * 100))
header = [
  'name',
  'total time (\u03BCs)',
  'mean (\u03BCs)',
  'stdev \u03C3',
  'samples',
  'total time (base)',
  'mean (base)',
  'stdev (base)',
  'samples (base)',
  'diff \u0394',
  f'{confidence_pct}% CI (\u0394)',
  '',
]
print(tab.tabulate(table, header, tablefmt="github"))