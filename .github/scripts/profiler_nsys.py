import argparse
import csv
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
args = parser.parse_args()

kernel_list = []
with open(args.input) as csv_file:
  csv_reader = csv.reader(csv_file)
  next(csv_reader)
  next(csv_reader)
  next(csv_reader)
  for row in csv_reader:
    if row:
      full_name = row[8]
      count = int(row[2])
      mean = float(row[3])
      stdev = float(row[7])
      if len(full_name) > 5 and full_name[:5] == "krnl_":
        name = full_name[5:]
        kernel_list.append([name, mean, stdev, count])

kernel_list.sort(key = lambda row: row[0])

data = [["name", "mean", "stdev", "count"]]
data += kernel_list

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
