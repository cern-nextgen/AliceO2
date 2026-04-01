import argparse
import csv
import tabulate as tab

parser = argparse.ArgumentParser()
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

def get_emoji(d, stdev):
  z = 1.96 # 95% confidence interval
  if d < -z * stdev:
    return ':green_circle:'
  elif d > z * stdev:
    return ':red_circle:'
  else:
    return ':white_circle:'

table = []
for baseline, current in zip(table_baseline, table_current):
  baseline_name, baseline_mean, _ = baseline
  name, mean, stdev = current
  assert(baseline_name == name)
  diff = baseline_mean - mean
  impact = 0.0 if stdev == 0.0 else diff / stdev
  emoji = get_emoji(diff, stdev)
  table.append([name, int(mean), f'{stdev:.2f}', int(diff), f'{impact:.2f}', emoji])

header = ['name', 'mean (\u03BCs)', 'stdev \u03C3', 'diff \u0394', '\u0394 / \u03C3', '']
print(tab.tabulate(table, header, tablefmt="github"))
