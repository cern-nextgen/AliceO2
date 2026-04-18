import argparse
import csv
import math
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--discard', type=int, default=0, help='Number of initial measurements to discard')
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

time_dict = dict({})
count_dict = dict({})
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  for row in csv_reader:
    name = row[2]
    time = float(row[3])
    count = row[1]
    if name in time_dict.keys():
      time_dict[name].append(time)
    else:
      time_dict[name] = [time]
      count_dict[name] = 1 if count == '' else int(count)

data = [["name", "mean", "stdev", "count"]]
for name, time_list in time_dict.items():
  count = count_dict[name]
  mean = statistics.mean(time_list[args.discard:]) / count
  runs = len(time_list[args.discard:])
  stdev = 0.0 if runs == 1 else statistics.stdev(time_list[args.discard:]) / math.sqrt(count)
  data.append([name, mean, stdev, runs * count])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
