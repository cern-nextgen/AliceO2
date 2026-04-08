import argparse
import csv
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

kernel_dict = dict({})
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  for row in csv_reader:
    full_name = row[13]
    time = (int(row[15]) - int(row[14])) / 1000.0
    if len(full_name) > 5 and full_name[:5] == "krnl_":
      name = full_name[5:-3]
      if name in kernel_dict.keys():
        kernel_dict[name].append(time)
      else:
        kernel_dict[name] = [time]

data = [["name", "mean", "stdev", "count"]]
for name, time_list in kernel_dict.items():
  count = len(time_list)
  mean = statistics.mean(time_list)
  stdev = 0 if count == 1 else statistics.stdev(time_list)
  data.append([name, mean, stdev, count])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
