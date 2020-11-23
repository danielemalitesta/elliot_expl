# http://jmcauley.ucsd.edu/data/tradesy/

import gzip
import argparse
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from config.configs import *

from operator import is_not
from functools import partial

parser = argparse.ArgumentParser(description="Run create url.")
parser.add_argument('--file_review', type=str,
                    default='reviews_tradesy.json.gz', help='Reviews')
parser.add_argument('--file_meta', type=str,
                    default='meta_tradesy_item_urls.json.gz', help='Metadata')
parser.add_argument('--dataset', nargs='?', default='tradesy', help='dataset name')
parser.add_argument('--k_core', type=int, default=5, help='K-core')

args = parser.parse_args()

g_reviews = gzip.open(data_path.format(args.dataset) + args.file_review, 'rb')
g_meta = gzip.open(data_path.format(args.dataset) + args.file_meta, 'rb')


# TRADESY 2014
def get_users_items(record):
    d = eval(record)
    if len(d.get('lists').get('bought') + d.get('lists').get('want')) > 0:
        current_user_items = []
        for b in d.get('lists').get('bought'):
            current_user_items.append({
                'USER': d.get('uid'),
                'ITEM': b
            })
        for w in d.get('lists').get('want'):
            current_user_items.append({
                'USER': d.get('uid'),
                'ITEM': w
            })
        return current_user_items


def get_urls(record):
    d = eval(record)
    if d.get('imageUrl'):
        return {
            'ITEM': d.get('itemID'),
            'URL': d.get('imageUrl')
        }


pool = Pool(cpu_count())
users_items = pool.map(get_users_items, g_reviews)
pool.close()
pool.join()
users_items = list(filter(partial(is_not, None), users_items))
users_items = [item for sublist in users_items for item in sublist]
df_users_items = pd.DataFrame(users_items)
del users_items

pool = Pool(cpu_count())
images = pool.map(get_urls, g_meta)
pool.close()
pool.join()
images = list(filter(partial(is_not, None), images))
df_images = pd.DataFrame(images)
del images

df_users_items = pd.merge(df_users_items, df_images, how='inner', on='ITEM').drop_duplicates()

# get top-k items
count = df_users_items.groupby('ITEM').size().reset_index(name='counts')
count = count.sort_values(by='counts', ascending=False)
top_k = count.iloc[:50000]  # in case several urls are not available
filtered_df_urm = pd.merge(df_users_items, top_k, on='ITEM', how='inner')
filtered_df_urm.drop('counts', axis='columns', inplace=True)

# k-core filtering
count_user = filtered_df_urm.groupby('USER').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)
count_user = count_user[count_user['counts'] >= args.k_core]
filtered_k_core = pd.merge(filtered_df_urm, count_user, on='USER', how='inner')
filtered_k_core.drop('counts', axis='columns', inplace=True)

final_users_items = pd.concat([pd.Series(filtered_k_core['USER']),
                               pd.Series(filtered_k_core['ITEM'])], axis=1)
final_users_items.columns = ['USER', 'ITEM']

final_users_items.to_csv(data_path.format(args.dataset) + 'all.tsv', index=False)

images_url = pd.concat([pd.Series(filtered_k_core['ITEM']),
                        pd.Series(filtered_k_core['URL'])], axis=1).drop_duplicates()

print('Statistics (before downloading images):')
print(f'''Lowest number of positive items per user: {args.k_core}''')
print(f'''Users: {len(final_users_items['USER'].unique())}''')
print(f'''Items: {len(final_users_items['ITEM'].unique())}''')
print(f'''Interactions: {len(final_users_items)}''')
print(f'''Sparsity: {1 - (len(final_users_items) / (len(final_users_items['USER'].unique()) * 
                                                  len(final_users_items['ITEM'].unique())))}''')

images_url.to_csv(all_items.format(args.dataset), index=False)
