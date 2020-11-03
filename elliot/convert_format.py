import argparse
import csv

parser = argparse.ArgumentParser(description="Run Dataset Format Conversion.")
parser.add_argument('--dataset', nargs='?', default='gowalla', help='dataset name: movielens, lastfm')

args = parser.parse_args()

df_train, df_test = [], []

with open('../data/{0}/train.txt'.format(args.dataset), "r") as f:
    line = f.readline()
    while line is not None and line != "":
        line = line.split('\n')
        line = line[0].split(' ')
        for lin in line[1:]:
            df_train.append([int(line[0]), int(lin), 1.0])
        line = f.readline()

with open('../data/{0}/test.txt'.format(args.dataset), "r") as f:
    line = f.readline()
    while line is not None and line != "":
        line = line.split('\n')
        line = line[0].split(' ')
        for lin in line[1:]:
            if lin == '':
                break
            else:
                df_test.append([int(line[0]), int(lin), 1.0])
        line = f.readline()

with open("../data/{0}/trainingset.tsv".format(args.dataset), "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(df_train)

with open("../data/{0}/testset.tsv".format(args.dataset), "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(df_test)
