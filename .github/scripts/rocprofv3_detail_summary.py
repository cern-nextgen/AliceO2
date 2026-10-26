# python3 rocprofv3_detail_summary.py --input amd-w7900_detail_counter_collection.csv --output amd-w7900_detail.csv

import argparse
import csv
import statistics
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='Raw rocprofv3 counter_collection CSV')
parser.add_argument('-o', '--output', required=True, help='Output CSV, averaged per (Kernel Name, Counter Name)')
args = parser.parse_args()

values = defaultdict(list)
launch_config = {}
with open(args.input) as csv_file:
  for row in csv.DictReader(csv_file):
    kernel = row['Kernel_Name']
    counter = row['Counter_Name']
    try:
      values[(kernel, counter)].append(float(row['Counter_Value']))
    except ValueError:
      continue
    if kernel not in launch_config:
      launch_config[kernel] = (
        row.get('Workgroup_Size', ''),
        row.get('Grid_Size', ''),
        row.get('VGPR_Count', ''),
        row.get('SGPR_Count', ''),
        row.get('LDS_Block_Size', ''),
        row.get('Scratch_Size', ''),
      )

data = [['Kernel Name', 'Block Size', 'Grid Size', 'VGPR Count', 'SGPR Count', 'LDS Block Size', 'Scratch Size',
         'Counter Name', 'Minimum', 'Maximum', 'Average', 'Stdev', 'Invocations']]
for (kernel, counter), vals in values.items():
  block_size, grid_size, vgpr_count, sgpr_count, lds_block_size, scratch_size = launch_config.get(kernel, ('',) * 6)
  data.append([
    kernel,
    block_size,
    grid_size,
    vgpr_count,
    sgpr_count,
    lds_block_size,
    scratch_size,
    counter,
    min(vals),
    max(vals),
    statistics.mean(vals),
    0.0 if len(vals) < 2 else statistics.stdev(vals),
    len(vals),
  ])

with open(args.output, 'w') as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerows(data)
