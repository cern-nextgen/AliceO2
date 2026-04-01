import argparse
import csv
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs', type=int, required=True, help='Number of runs')
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

time_dict = dict({})
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  for row in csv_reader:
    full_name = row[13]
    time = (int(row[15]) - int(row[14])) / 1000.0
    if len(full_name) > 5 and full_name[:5] == "krnl_":
      name = full_name[5:-3]
      if name in time_dict.keys():
        time_dict[name].append(time)
      else:
        time_dict[name] = [time]

data = [["name", "time", "stdev"]]
for name, time_list in time_dict.items():
  count = len(time_list) / args.runs
  mean = int(statistics.mean(time_list) * count)
  stdev = 0 if args.runs == 1 else statistics.stdev(time_list) * count
  data.append([name, mean, stdev])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
