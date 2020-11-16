from config.configs import *
import pandas as pd
import shutil
import os

downloaded_images = os.listdir(images_path.format('pinterest'))
downloaded_images.sort(key=lambda x: int(x.split(".")[0]))
downloaded_images = [int(f[:-4]) for f in downloaded_images]

downloaded = pd.DataFrame(downloaded_images, columns=['ID'])

all_images = pd.read_csv(all_items.format('pinterest'))
all_images.columns = ['ID', 'url']
available_images = pd.merge(all_images, downloaded, on='ID', how='inner')

all_user_item = pd.read_csv(data_path.format('pinterest') + 'all.tsv', sep='\t')
print(f'Number of interactions before filtering: {len(all_user_item)}')
available_user_item = pd.merge(all_user_item, available_images, on='url', how='inner')
print(f'Number of interactions after filtering: {len(available_user_item)}\n\n')

available_user_item['USER_ID'] = available_user_item.groupby('user').grouper.group_info[0]
available_user_item['ITEM_ID'] = available_user_item.groupby('url').grouper.group_info[0]
available_user_item = available_user_item.sort_values(by='USER_ID')

users_items = pd.concat([available_user_item['USER_ID'], available_user_item['ITEM_ID']], axis=1)
users_items.to_csv(all_interactions.format('pinterest'), index=False, header=False, sep='\t')

available_user_item = available_user_item.reset_index(drop=True)
users_df = pd.concat([pd.Series(available_user_item['user'].unique()),
                      pd.Series(available_user_item['USER_ID'].unique())], axis=1)
items_df = pd.concat([pd.Series(available_user_item['url'].unique()),
                      pd.Series(available_user_item['ITEM_ID'].unique())], axis=1).sort_values(by=1)
users_df.to_csv(users.format('pinterest'), index=False, header=False, sep='\t')
items_df.to_csv(items.format('pinterest'), index=False, header=False, sep='\t')

# check k-core
count_user = available_user_item.groupby('USER_ID').size().reset_index(name='counts')
count_user = count_user.sort_values(by='counts', ascending=False)
print(f'''Lowest number of positive items per user: {count_user.iloc[-1, 1]}''')

print('Statistics (after downloading images):')
print(f'''Users: {len(available_user_item['USER_ID'].unique())}''')
print(f'''Items: {len(available_user_item['ITEM_ID'].unique())}''')
print(f'''Interactions: {len(available_user_item)}''')
print(f'''Sparsity: {1 - (len(available_user_item) / (len(available_user_item['USER_ID'].unique()) * 
                                                      len(available_user_item['ITEM_ID'].unique())))}''')

items_df.columns = ['url', 'NEW_ID']
new_image_mapping = pd.merge(available_images, items_df, on='url', how='inner')

for i, row in new_image_mapping.iterrows():
    shutil.move(
        images_path.format('pinterest') + str(row['ID']) + '.jpg',
        images_path.format('pinterest') + str(row['NEW_ID']) + '_new.jpg'
    )

for i, row in new_image_mapping.iterrows():
    shutil.move(
        images_path.format('pinterest') + str(row['NEW_ID']) + '_new.jpg',
        images_path.format('pinterest') + str(row['NEW_ID']) + '.jpg'
    )
