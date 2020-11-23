# DATASET
data_path = '../data/{0}/'
imagenet_classes_path = '../data/imagenet_classes.txt'
all_items = data_path + 'all_items.csv'
all_interactions = data_path + 'all_interactions.tsv'
users = data_path + 'users.tsv'
items = data_path + 'items.tsv'
training_path = data_path + 'trainingset.tsv'
validation_path = data_path + 'validationset.tsv'
test_path = data_path + 'testset.tsv'
original = data_path + 'original/'
images_path = original + 'images/'
classes_path = original + 'classes_{0}.csv'
features_path = original + 'features_{0}_{1}.npy'

# RESULTS
weight_dir = '../results/rec_model_weights'
results_dir = '../results/rec_results'
