import argparse
import csv
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runs', type=int, required=True, help='Number of runs')
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

ntsi_list = []
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  next(csv_reader)
  next(csv_reader)
  for row in csv_reader:
    if row:
      full_name = row[8]
      instances = int(row[2])
      time = float(row[3])
      sigma = float(row[7])
      if len(full_name) > 5 and full_name[:5] == "krnl_":
        name = full_name[5:]
        ntsi_list.append([name, time, sigma, instances])

ntsi_list.sort(key = lambda row: row[0])

data = [["name", "time", "stdev"]]
for name, time, sigma, instances in ntsi_list:
  count = instances / args.runs
  mean = int(time * count)
  stdev = sigma * count
  data.append([name, mean, stdev])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
