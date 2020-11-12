from sklearn.model_selection import train_test_split
from config.configs import *
import pandas as pd
import argparse
import random

random.seed(1234)

parser = argparse.ArgumentParser(description="Run dataset splitting.")
parser.add_argument('--dataset', nargs='?', default='amazon_fashion', help='dataset name')
parser.add_argument('--test_size', type=float, default=0.2, help='test set size')
parser.add_argument('--percentage', type=bool, default=False, help='whether using a % to express val e test set')
parser.add_argument('--validation_size', type=float, default=0.1, help='validation set size')
parser.add_argument('--shuffle', type=bool, default=True, help='dataset shuffle')
parser.add_argument('--column_stratify', type=list, default=[0], help='list of columns to use for stratification')
args = parser.parse_args()

# read all interactions
df = pd.read_csv(all_interactions.format(args.dataset), delimiter='\t', header=None)

num_users, num_items = df[0].nunique(), df[1].nunique()

interactions = df.copy()
interactions = interactions.sample(len(interactions)).reset_index(drop=True)

validation, test = [], []

for u in range(num_users):
    current_item = interactions[interactions[0] == u].iloc[0, 1]
    index = interactions[interactions[0] == u].iloc[0].name
    validation.append({
        0: u,
        1: current_item
    })
    interactions.drop(index=index, inplace=True)

for u in range(num_users):
    current_item = interactions[interactions[0] == u].iloc[0, 1]
    index = interactions[interactions[0] == u].iloc[0].name
    test.append({
        0: u,
        1: current_item
    })
    interactions.drop(index=index, inplace=True)

validation = pd.DataFrame(validation)
test = pd.DataFrame(test)

# # split train/test
# train, test = train_test_split(df,
#                                test_size=args.test_size if args.percentage else num_users,
#                                random_state=1234,
#                                shuffle=args.shuffle,
#                                stratify=df[args.column_stratify])
#
# # split train/validation
# train, validation = train_test_split(train,
#                                      test_size=args.validation_size if args.percentage else num_users,
#                                      random_state=1234,
#                                      shuffle=args.shuffle,
#                                      stratify=train[args.column_stratify])

# sort according to user
train = interactions.sort_values(by=args.column_stratify)
validation = validation.sort_values(by=args.column_stratify)
test = test.sort_values(by=args.column_stratify)
train[2] = [1.0] * len(train)
validation[2] = [1.0] * len(validation)
test[2] = [1.0] * len(test)

# write to file
train.to_csv(training_path.format(args.dataset), index=False, sep='\t', header=None)
validation.to_csv(validation_path.format(args.dataset), index=False, sep='\t', header=None)
test.to_csv(test_path.format(args.dataset), index=False, sep='\t', header=None)
