import gzip
import json
import csv
import argparse
from config.configs import *

images = []

parser = argparse.ArgumentParser(description="Run create url.")
parser.add_argument('--file', type=str, default='meta_AMAZON_FASHION.json.gz', help='Metadata path')
parser.add_argument('--dataset', nargs='?', default='amazon_fashion', help='dataset name')

args = parser.parse_args()


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    for d in parse(path):
        if d.get('image'):
            images.append({
                'ASIN': d.get('asin'),
                'URL': d.get('image')[-1]
            })


getDF(data_path.format(args.dataset) + args.file)

keys = images[0].keys()
with open('../data/{}/all_items.csv'.format(args.dataset), 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(images)
