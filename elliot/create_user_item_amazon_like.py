import os
import gzip
import json
import pandas as pd
import argparse
import shutil
import sys
import math
from config.configs import *

parser = argparse.ArgumentParser(description="Run create url.")
parser.add_argument('--file', type=str, default='AMAZON_FASHION.json.gz', help='Metadata path')
parser.add_argument('--dataset', nargs='?', default='amazon_fashion', help='dataset name')
parser.add_argument('--core', type=int, default=5, help='k-core')

args = parser.parse_args()

user_item = []


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    for d in parse(path):
        if d.get('reviewerID') and d.get('asin'):
            user_item.append({
                'USER': d.get('reviewerID'),
                'ASIN': d.get('asin')
            })


getDF(data_path.format(args.dataset) + args.file)
user_item_df = pd.DataFrame(user_item)  # all users and items

downloaded_images = os.listdir(images_path.format(args.dataset))  # available images (no duplicates no 404 codes)
downloaded_images = [file.split('.')[0] for file in downloaded_images]
downloaded = pd.DataFrame(downloaded_images, columns=['ASIN'])
downloaded['Downloaded'] = pd.Series([True] * len(downloaded))

all_images = pd.read_csv(all_items.format(args.dataset))  # all items

all_images = pd.merge(all_images, downloaded, on='ASIN', how='outer')
# all items, NaN where we have either duplicates or 404 codes

available_images = all_images.groupby('URL').filter(lambda g: g.isnull().values.sum() < len(g))
# filter out 404 codes from all interactions

for i, row in available_images.iterrows():
    if math.isnan(row['Downloaded']):
        source_file = images_path.format(args.dataset) + \
                      str(available_images[(available_images['URL'] == row['URL']) &
                                           (available_images['Downloaded'])]['ASIN'].iloc[0]) + '.jpg'
        dest_file = images_path.format(args.dataset) + str(row['ASIN']) + '.jpg'
        shutil.copy(source_file, dest_file)

available_interactions = pd.merge(user_item_df, available_images, on='ASIN', how='inner')
# True images which are not in the URM are filtered out

filtered_k_core = available_interactions.groupby(by='USER').filter(lambda g: len(g) >= args.core)
# after k-core, some True images might be dropped, so there exist NaN images (i.e., duplicates)
# which don't have any original image anymore, but they must remain

filtered_k_core['USER_ID'] = filtered_k_core.groupby('USER').grouper.group_info[0]
filtered_k_core['ITEM_ID'] = filtered_k_core.groupby('ASIN').grouper.group_info[0]
filtered_k_core = filtered_k_core.sort_values(by='USER_ID')

num_users, num_items = filtered_k_core['USER_ID'].nunique(), filtered_k_core['ITEM_ID'].nunique()
print(f'Users: {num_users}, Items: {num_items}, Interactions: {len(filtered_k_core)}')
filtered_k_core.to_csv(data_path.format(args.dataset) + 'all.tsv', index=False, sep='\t')

users_items = pd.concat([filtered_k_core['USER_ID'], filtered_k_core['ITEM_ID']], axis=1)
users_items.to_csv(all_interactions.format(args.dataset), index=False, header=False, sep='\t')

filtered_k_core = filtered_k_core.reset_index(drop=True)
users_df = pd.concat([pd.Series(filtered_k_core['USER'].unique()),
                      pd.Series(filtered_k_core['USER_ID'].unique())], axis=1)
items_df = pd.concat([pd.Series(filtered_k_core['ASIN'].unique()),
                      pd.Series(filtered_k_core['ITEM_ID'].unique())], axis=1).sort_values(by=1)
users_df.to_csv(users.format(args.dataset), index=False, header=False, sep='\t')
items_df.to_csv(items.format(args.dataset), index=False, header=False, sep='\t')

for i, row in items_df.iterrows():
    source_file = images_path.format(args.dataset) + str(row[0]) + '.jpg'
    dest_file = images_path.format(args.dataset) + str(row[1]) + '.jpg'
    shutil.move(source_file, dest_file)
    sys.stdout.write('\r%d/%d samples completed' % (i + 1, len(filtered_k_core)))
    sys.stdout.flush()

# here, we remove True images which were not in the URM and the ones that were dropped after k-core
listdir = os.listdir(images_path.format(args.dataset))
final_images = [str(i) for i in range(num_items)]

for file in listdir:
    if file.split('.')[0] not in final_images:
        os.remove(images_path.format(args.dataset) + file)
