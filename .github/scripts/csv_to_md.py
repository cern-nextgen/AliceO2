import argparse
import csv
import math
import tabulate as tab

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs', type=int, required=True, help='Number of runs')
parser.add_argument('-b', '--baseline', required=True, help='Baseline CSV file')
parser.add_argument('-c', '--current', required=True, help='Current CSV file')
args = parser.parse_args()

def get_2d_list(csv_filename):
  with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    return [[str(name), float(mean), float(stdev), int(count)] for name, mean, stdev, count in csv_reader]

table_baseline = get_2d_list(args.baseline)
table_current = get_2d_list(args.current)

def student(x, sx, m, y, sy, n):
  s = 0.0 if m < 2 or n < 2 else math.sqrt(((m - 1) * sx**2 + (n - 1) * sy**2) / (m + n - 2))
  d = x - y
  t = 0.0 if s == 0.0 else math.sqrt((n * m) / (n + m)) * d / s
  return d, s, t

def get_emoji(t):
  quantile = 2.0 # 95% confidence interval
  if t < -quantile:
    return ':green_circle:'
  elif t > quantile:
    return ':red_circle:'
  else:
    return ':white_circle:'

table = []
for baseline, current in zip(table_baseline, table_current):
  baseline_name, baseline_mean, baseline_stdev, count_baseline = baseline
  name, mean, stdev, count = current
  assert(baseline_name == name)
  total_time = mean * (count // args.runs)
  d, s, t = student(baseline_mean, baseline_stdev, args.runs, mean, stdev, args.runs)
  emoji = get_emoji(t)
  table.append([name, int(total_time), int(mean), f'{stdev:.2f}', int(baseline_mean), f'{baseline_stdev:.2f}', f'{d:.2f}', f'{t:.2f}', emoji])

header = ['name', 'total time (\u03BCs)', 'mean (\u03BCs)', 'stdev \u03C3', 'mean (old)', 'stdev (old)', 'diff \u0394', 't', '']
print(tab.tabulate(table, header, tablefmt="github"))
