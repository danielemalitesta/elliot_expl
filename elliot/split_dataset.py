from sklearn.model_selection import train_test_split
from config.configs import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Run dataset splitting.")
parser.add_argument('--dataset', nargs='?', default='gowalla', help='dataset name')
parser.add_argument('--test_size', type=float, default=0.2, help='test set size')
parser.add_argument('--validation_size', type=float, default=0.1, help='validation set size')
parser.add_argument('--shuffle', type=bool, default=True, help='dataset shuffle')
parser.add_argument('--column_stratify', type=list, default=[0], help='list of columns to use for stratification')
args = parser.parse_args()

# read all interactions
df = pd.read_csv(all_interactions.format(args.dataset), delimiter='\t', header=None)

# split train/test
train, test = train_test_split(df,
                               test_size=args.test_size,
                               random_state=1234,
                               shuffle=args.shuffle,
                               stratify=df[args.column_stratify])

# split train/validation
train, validation = train_test_split(train,
                                     test_size=args.validation_size,
                                     random_state=1234,
                                     shuffle=args.shuffle,
                                     stratify=train[args.column_stratify])

# sort according to user
train = train.sort_values(by=args.column_stratify)
validation = validation.sort_values(by=args.column_stratify)
test = test.sort_values(by=args.column_stratify)

# write to file
train.to_csv(training_path.format(args.dataset), index=False, sep='\t', header=None)
validation.to_csv(validation_path.format(args.dataset), index=False, sep='\t', header=None)
test.to_csv(test_path.format(args.dataset), index=False, sep='\t', header=None)
