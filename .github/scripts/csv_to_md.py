import sys
import csv
import tabulate as tab

csv_benchmark = sys.argv[1]
csv_baseline = sys.argv[2]

def html_inline(x):
  if x == 0:
    color = "black"
    x = "{:.1f}".format(x)
  elif x < 0:
    color = "green"
    x = "{:.1f}".format(x)
  elif x > 0:
    color = "red"
    x = "+{:.1f}".format(x)
  return f"<span style=\"color:{color}\">{x}</span>"

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
  assert(benchmark[0] == baseline[0] and benchmark[2:4] == baseline[2:4])
  name = benchmark[0]
  time = benchmark[1]
  d = float(baseline[1]) - float(benchmark[1])
  difference = html_inline(d)
  percent = html_inline(100 * d / float(baseline[1]))
  count = benchmark[2]
  events = benchmark[3]
  table.append([name, time, difference, percent, count, events])

header = ["name", "time", "difference", "percent", "count", "events"]
print(tab.tabulate(table, header, tablefmt="github"))
