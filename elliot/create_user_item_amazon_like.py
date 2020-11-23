import os
import pandas as pd
import argparse
import shutil
import sys
import math
import cv2
from config.configs import *

parser = argparse.ArgumentParser(description="Run create users-items.")
parser.add_argument('--dataset', nargs='?', default='amazon_clothing', help='dataset name')

args = parser.parse_args()

downloaded_images = os.listdir(images_path.format(args.dataset))  # available images (no duplicates no 404 codes)
downloaded_images = [file.split('.')[0] for file in downloaded_images]
downloaded = pd.DataFrame(downloaded_images, columns=['ASIN'])
downloaded['Downloaded'] = pd.Series([True] * len(downloaded))

no_rgb_images = []

for index, row in downloaded.iterrows():
    # if image is not RGB, it's not suitable for recommendation
    if len(cv2.imread(images_path.format(args.dataset) + str(row['ASIN']) + '.jpg', -1).shape) == 2:
        no_rgb_images.append(row['ASIN'])

print('List of no RGB images:')
print(no_rgb_images)
if len(no_rgb_images) > 0:
    exit(1)

all_images = pd.read_csv(all_items.format(args.dataset))  # all items

all_images = pd.merge(all_images, downloaded, on='ASIN', how='outer')
# all items, NaN where we have either duplicates or 404 codes

available_images = all_images.groupby('URL').filter(lambda g: g.isnull().values.sum() < len(g))
# filter out 404 codes from all images

for i, row in available_images.iterrows():
    if math.isnan(row['Downloaded']):
        source_file = images_path.format(args.dataset) + \
                      str(available_images[(available_images['URL'] == row['URL']) &
                                           (available_images['Downloaded'])]['ASIN'].iloc[0]) + '.jpg'
        dest_file = images_path.format(args.dataset) + str(row['ASIN']) + '.jpg'
        shutil.copy(source_file, dest_file)

df_users_items = pd.read_csv(data_path.format(args.dataset) + 'all.tsv')
available_interactions = pd.merge(df_users_items, available_images, on='ASIN', how='inner')
# I'm dropping 404 images

# filtered_k_core = available_interactions.groupby(by='USER').filter(lambda g: len(g) >= args.core)
# after k-core, some True images might be dropped, so there exist NaN images (i.e., duplicates)
# which don't have any original image anymore, but they must remain

available_interactions['USER_ID'] = available_interactions.groupby('USER').grouper.group_info[0]
available_interactions['ITEM_ID'] = available_interactions.groupby('ASIN').grouper.group_info[0]
available_interactions = available_interactions.sort_values(by='USER_ID')

# check k-core
count_user = available_interactions.groupby('USER_ID').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)

print('Statistics (after downloading images):')
print(f'''Lowest number of positive items per user: {count_user.iloc[-1, 1]}''')
print(f'''Users: {len(available_interactions['USER'].unique())}''')
print(f'''Items: {len(available_interactions['ASIN'].unique())}''')
print(f'''Interactions: {len(available_interactions)}''')
print(f'''Sparsity: {1 - (len(available_interactions) / (len(available_interactions['USER'].unique()) * 
                                                  len(available_interactions['ASIN'].unique())))}''')

available_interactions.to_csv(data_path.format(args.dataset) + 'all.tsv', index=False, sep='\t')

users_items = pd.concat([available_interactions['USER_ID'], available_interactions['ITEM_ID']], axis=1)
users_items.to_csv(all_interactions.format(args.dataset), index=False, header=False, sep='\t')

available_interactions = available_interactions.reset_index(drop=True)
users_df = pd.concat([pd.Series(available_interactions['USER'].unique()),
                      pd.Series(available_interactions['USER_ID'].unique())], axis=1)
items_df = pd.concat([pd.Series(available_interactions['ASIN'].unique()),
                      pd.Series(available_interactions['ITEM_ID'].unique())], axis=1).sort_values(by=1)
users_df.to_csv(users.format(args.dataset), index=False, header=False, sep='\t')
items_df.to_csv(items.format(args.dataset), index=False, header=False, sep='\t')

for i, row in items_df.iterrows():
    source_file = images_path.format(args.dataset) + str(row[0]) + '.jpg'
    dest_file = images_path.format(args.dataset) + str(row[1]) + '.jpg'
    shutil.move(source_file, dest_file)
    sys.stdout.write('\r%d/%d samples completed' % (i + 1, len(items_df)))
    sys.stdout.flush()

# here, we remove True images which were not in the URM and the ones that were dropped after k-core
# listdir = os.listdir(images_path.format(args.dataset))
# final_images = [str(i) for i in range(len(items_df))]
#
# for file in listdir:
#     if file.split('.')[0] not in final_images:
#         os.remove(images_path.format(args.dataset) + file)
