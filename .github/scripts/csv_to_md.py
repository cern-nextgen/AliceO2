import sys
import csv
import tabulate as tab

csv_benchmark = sys.argv[1]
csv_baseline = sys.argv[2]

pretty = lambda x : "{:.1f}".format(x) if x <= 0 else "+{:.1f}".format(x)

with open(csv_benchmark) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  table_benchmark = [row for row in csv_reader]
 
with open(csv_baseline) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  table_baseline = [row for row in csv_reader]

table = []
for benchmark, baseline in zip(table_benchmark, table_baseline):
  assert(benchmark[0] == baseline[0])
  name = benchmark[0]
  time = benchmark[1]
  stdev = benchmark[1]
  d = float(baseline[1]) - float(benchmark[1])
  emoji = ':red_circle:' if 0 < d else ':green_circle:'
  difference = pretty(d)
  percent = pretty(100 * d / float(baseline[1]))
  table.append([name, time, stdev, emoji, difference, percent])

header = ["name", "time", "stdev", "", "difference", "percent"]
print(tab.tabulate(table, header, tablefmt="github"))
