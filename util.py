import pandas as pd
import os
import tensorflow as tf
import numpy as np
import random
from pathlib import Path
from sklearn.utils import class_weight


def readDir(direc):
    data = []
    normal_cases_dir = direc / 'NORMAL'
    pneumonia_cases_dir = direc / 'PNEUMONIA'

    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

    for img in normal_cases:
        data.append((str(img), '0'))
    for img in pneumonia_cases:
        data.append((str(img), '1'))

    return data


def readData(direc):
    data_dir = Path(direc)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    data = readDir(train_dir) + readDir(val_dir) + readDir(test_dir)
    data = pd.DataFrame(data, columns=['image', 'label'], index=None)
    data = data.sample(frac=1.0).reset_index(drop=True)

    return data


def setSeed(seed):
    tf.keras.backend.clear_session()
    seed_value = seed

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # os.environ['TF_DETERMINISTIC_OPS']=str(1)

    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def splitTrainValTest(data, train_cutoff, val_cutoff):
    splits = np.cumsum([train_cutoff, val_cutoff])

    n = len(data)
    train_cutoff = int(splits[0] * n)
    train_data = data[:train_cutoff].sample(frac=1.0).reset_index(drop=True)

    val_cutoff = int(splits[1] * n)
    val_data = data[train_cutoff:val_cutoff].sample(frac=1.0).reset_index(drop=True)

    test_data = data[val_cutoff:].sample(frac=1.0).reset_index(drop=True)

    return (train_data, val_data, test_data)


def getClassWeights(train_data):
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_data['label']), train_data['label']
    )

    return class_weights
