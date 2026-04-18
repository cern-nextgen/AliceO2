import argparse
import csv
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--discard', type=int, default=0, help='Number of initial measurements to discard')
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

time_dict = dict({})
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  for name, time, _, _ in csv_reader:
    if name in time_dict.keys():
      time_dict[name].append(float(time))
    else:
      time_dict[name] = [float(time)]

data = [["name", "time", "stdev"]]
for name, time_list in time_dict.items():
  mean = statistics.mean(time_list[args.discard:])
  stdev = statistics.stdev(time_list[args.discard:])
  data.append([name, mean, stdev])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
