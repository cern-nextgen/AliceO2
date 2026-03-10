import sys
import csv
import tabulate as tab

csv_filename = sys.argv[1]

with open(csv_filename) as csv_file:
  csv_reader = csv.reader(csv_file)
  header = next(csv_reader)
  table = [row for row in csv_reader]

print(tab.tabulate(table, header, tablefmt="github"))
