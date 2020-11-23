from vision.FeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *
from utils.read import *
import pandas as pd
import argparse
import numpy as np
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_fashion', help='dataset path')
    parser.add_argument('--model_name', nargs='?', default='ResNet152', help='model for feature extraction')
    parser.add_argument('--output_name', nargs='?', default='conv5_block3_out', help='output layer name')

    return parser.parse_args()


def classify_extract():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # model setting
    model = FeatureExtractor(
        model_name=args.model_name,
        output_layer=args.output_name,
        imagenet=read_imagenet_classes_txt(imagenet_classes_path)
    )

    # dataset setting
    data = Dataset(
        dataset=args.dataset,
        resize=(224, 224),
        model_name=args.model_name
    )
    print('Loaded dataset from %s' % images_path.format(args.dataset))

    # features and classes
    df = pd.DataFrame([], columns={'ImageID', 'ClassStr', 'ClassNum', 'Prob'})
    features_shape = [data.num_samples, *model.model.output.shape[1:]]
    features = np.empty(shape=features_shape)

    # classification and features extraction
    print('Starting classification...\n')
    start = time.time()

    for i, d in enumerate(data):
        image, path = d
        out_class = model.classify(sample=(image, path))
        features[i] = model.model(image)
        df = df.append(out_class, ignore_index=True)
        if (i + 1) % 100 == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()

    write_csv(df=df, filename=classes_path.format(args.dataset, args.model_name.lower()))
    save_np(npy=features, filename=features_path.format(args.dataset, args.model_name.lower(), args.output_name))

    end = time.time()

    print('\n\nClassification and feature extraction completed in %f seconds.' % (end - start))
    print('Saved features numpy in ==> %s' % features_path.format(args.dataset))
    print('Saved classification file in ==> %s' % classes_path.format(args.dataset))


if __name__ == '__main__':
    classify_extract()
