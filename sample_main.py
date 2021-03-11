from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--type_experiment', type=str, default='baselines')
parser.add_argument('--dataset', type=str, default='amazon_men')
args = parser.parse_args()

run_experiment("config_files/{0}_{1}.yml".format(args.type_experiment, args.dataset))
