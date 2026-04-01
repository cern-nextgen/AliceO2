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
    return [[str(name), float(mean), float(stdev)] for name, mean, stdev in csv_reader]

table_baseline = get_2d_list(args.baseline)
table_current = get_2d_list(args.current)

def student(meanX, stdevX, runsX, meanY, stdevY, runsY):
  s2 = ((runsX - 1) * stdevX**2 + (runsY - 1) * stdevY**2) / (runsX + runsY - 2)
  return (meanX - meanY) / math.sqrt(s2 / runsX + s2 / runsY) if s2 > 0.0 else 0.0

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
  baseline_name, baseline_mean, baseline_stdev = baseline
  name, mean, stdev = current
  assert(baseline_name == name)
  diff = baseline_mean - mean
  t = student(baseline_mean, baseline_stdev, args.runs, mean, stdev, args.runs) if args.runs > 2 else 0.0
  emoji = get_emoji(t)
  table.append([name, int(mean), f'{stdev:.2f}', int(diff), f'{t:.2f}', emoji])

header = ['name', 'mean (\u03BCs)', 'stdev \u03C3', 'diff \u0394', 't', '']
print(tab.tabulate(table, header, tablefmt="github"))
