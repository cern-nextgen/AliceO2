import sys
import csv
import statistics

in_csv = sys.argv[1]
out_csv = sys.argv[2]

time_dict = dict({})

with open(in_csv) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  for name, time, _, _ in csv_reader:
    if name in time_dict.keys():
      time_dict[name].append(float(time))
    else:
      time_dict[name] = [float(time)]

data = [["name", "time", "stdev"]]
for name, time_list in time_dict.items():
  mean = statistics.mean(time_list)
  stdev = statistics.stdev(time_list)
  data.append([name, mean, stdev])

with open(out_csv, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
