import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="""Takes an embedding file in space-separated plain text, saving
it in Pandas-compatible HDF5 format
""")
parser.add_argument('plain_file', help="Prefix of embedding file")
parser.add_argument('hdf_file', help="Prefix of embedding file")
args = parser.parse_args()

words = []
values = []
for line in open(args.plain_file):
    parts = line.strip().split(" ")
    words.append(parts[0])
    values.append(map(float, parts[1:]))

D = pd.DataFrame(values, index=words)
D.to_hdf(args.hdf_file, 'embedding')
