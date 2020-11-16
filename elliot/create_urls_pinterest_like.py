from config.configs import *
import bson
import pandas as pd

with open('../data/pinterest/subset_iccv_board_pins.bson', 'rb') as f:
    users_pins = bson.decode_all(f.read())

with open('../data/pinterest/subset_iccv_pin_im.bson', 'rb') as f:
    im_per_pins = bson.decode_all(f.read())

print(f'Number of users (without filtering): {len(users_pins)}')
print(f'Number of interactions (without filtering): {len(im_per_pins)}')

users_items = [{'user': up['board_id'], 'items': up['pins']} for up in users_pins]
items_urls = [{'item': ipp['pin_id'], 'url': ipp['im_url']} for ipp in im_per_pins]
df_items_url = pd.DataFrame(items_urls)
urm = []

for inter in users_items:
    for it in inter['items']:
        urm.append({
            'user': inter['user'],
            'item': it
        })

df_urm = pd.DataFrame(urm)
df_urm = pd.merge(df_urm, df_items_url, on='item', how='outer')
del urm, df_items_url, im_per_pins, users_pins

df_urm.drop_duplicates(inplace=True, ignore_index=True)
df_urm.drop('item', axis='columns', inplace=True)

count = df_urm.groupby('url').size().reset_index(name='counts')
count = count.sort_values(by='counts', ascending=False)
top_k = count.iloc[:20000]  # in case several urls are not available
filtered_df_urm = pd.merge(df_urm, top_k, on='url', how='inner')
filtered_df_urm.drop('counts', axis='columns', inplace=True)

################################################################################
# k-core filtering
count_user = filtered_df_urm.groupby('user').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)
count_user = count_user[count_user['counts'] >= 5]
filtered_df_urm = pd.merge(filtered_df_urm, count_user, on='user', how='inner')
filtered_df_urm.drop('counts', axis='columns', inplace=True)

print(f'''Lowest number of positive items per user: {count_user.iloc[-1, 1]}''')
print('Statistics (before downloading images):')
print(f'''Users: {len(filtered_df_urm['user'].unique())}''')
print(f'''Items: {len(filtered_df_urm['url'].unique())}''')
print(f'''Interactions: {len(filtered_df_urm)}''')
print(f'''Sparsity: {1 - (len(filtered_df_urm) / (len(filtered_df_urm['user'].unique()) * 
                                                  len(filtered_df_urm['url'].unique())))}''')
################################################################################

filtered_df_urm.to_csv(data_path.format('pinterest') + 'all.tsv', index=False, sep='\t')

images = filtered_df_urm['url'].unique().tolist()

df_images = [{'ID': i, 'URL': im} for i, im in enumerate(images)]
df_images = pd.DataFrame(df_images)
df_images.to_csv(all_items.format('pinterest'), index=False)
