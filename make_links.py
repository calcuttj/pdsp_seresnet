import h5py as h5
from glob import glob as ls
import sys

filelist_name= sys.argv[1]
output_name = sys.argv[2]
#files = ls(ls_str)
#print(files)

with open(filelist_name, 'r') as filelist:
  lines = [l.strip() for l in filelist.readlines()]
  print(lines)

with h5.File(output_name, 'w') as f:
  i = 0
  for n in lines:
    print(n)
    f[f'link{i}'] = h5.ExternalLink(n, '/')
    i += 1
