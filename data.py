import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

def add_filepath():
    # add image filepath to the dataframe
    all_xray_df = pd.read_csv('input/chest_xrays/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('input', 'chest_xrays', 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    return all_xray_df

def visualizing_label_count(all_xray_df):
    # visualizing label counts
    label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
    fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
    ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts))+0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation = 90)

def onehot_encoding(all_xray_df):
    # change the labels to one hot encoded values
    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    from itertools import chain
    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

    # use only some of the columns
    save_columns = ['Image Index', 'Finding Labels', 'path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    updated_xray_df = all_xray_df.loc[:, save_columns]
    return updated_xray_df

def train_val_test_split(updated_xray_df):
    # split the dataset into 92% training, 5% validation and 3% test
    image_indexes = np.array(updated_xray_df['Image Index'])

    train_len = int(len(image_indexes) * 0.92)
    val_len = int(len(image_indexes) * 0.05)
    test_len = int(len(image_indexes) * 0.03)

    np.random.shuffle(image_indexes)
    train_indexes = image_indexes[:train_len]
    val_indexes = image_indexes[train_len: train_len + val_len]
    test_indexes = image_indexes[train_len + val_len: ]

    train_path = "input/chest_xrays/train.csv"
    val_path = "input/chest_xrays/val.csv"
    test_path = "input/chest_xrays/test.csv"

    if not os.path.exists(train_path):
        train_df = updated_xray_df[updated_xray_df['Image Index'].map(lambda x: True if x in train_indexes else False)]
        train_df.to_csv(train_path, index=False)
    if not os.path.exists(val_path):
        val_df = updated_xray_df[updated_xray_df['Image Index'].map(lambda x: True if x in val_indexes else False)]
        val_df.to_csv(val_path, index=False)
    if not os.path.exists(test_path):
        test_df = updated_xray_df[updated_xray_df['Image Index'].map(lambda x: True if x in test_indexes else False)]
        test_df.to_csv(test_path, index=False)

def main(visualizing_label_count=False):
    all_xray_df = add_filepath()
    if visualizing_label_count:
        visualizing_label_count()
    updated_xray_df = onehot_encoding(all_xray_df)
    train_val_test_split(updated_xray_df)

if __name__ == '__main__':
    main(True)
