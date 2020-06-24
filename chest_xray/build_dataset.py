# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob
from itertools import chain
from utils import get_class_counts
import config


def add_image_path(df, images_base_path):
    images_path = {os.path.basename(x): x for x in glob(
        os.path.join(images_base_path, '*.png'))}
    print('Scans found:', len(images_path), ', Total Headers', df.shape[0])
    df['Image Path'] = df['Image Index'].map(images_path.get)
    return df


def onehot_encode(df):
    df['Finding Labels'] = df['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))
    labels = np.unique(
        list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    labels = [x for x in labels if len(x) > 0]
    print('All Labels ({}): {}'.format(len(labels), labels))
    for label in labels:
        if len(label) > 1:  # leave out empty labels
            df[label] = df['Finding Labels'].map(
                lambda finding: 1 if label in finding else 0)
    return df


def visualize_class_count(df, class_names):
    _, class_count_dict = get_class_counts(df, class_names)
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.bar(np.arange(len(class_count_dict))+0.5, class_count_dict.values())
    ax.set_xticks(np.arange(len(class_count_dict))+0.5)
    _ = ax.set_xticklabels(class_count_dict.keys(), rotation=90)


def chest_xrays14():
    # add image filepath to the dataframe
    chest_xrays14_df = pd.read_csv(config.CHESTXRAY14_METADATA_PATH)
    chest_xrays14_df = add_image_path(
        chest_xrays14_df, config.CHESTXRAY14_IMAGES_BASE_PATH)

    # change the labels to one hot encoded values
    chest_xrays14_df = onehot_encode(chest_xrays14_df)

    # keep only some of the columns
    chest_xrays14_df = chest_xrays14_df.loc[:, config.CHESTXRAY14_COLS]

    return chest_xrays14_df


def TB_chest_xrays():
    # create TB_xray_shenzen dataframe and process it
    TB_shenzen_df = pd.read_csv(config.TB_SHENZHEN_METADATA_PATH)
    TB_shenzen_df = TB_shenzen_df.rename(
        columns={'study_id': 'Image Index', 'findings': 'Finding Labels'})
    TB_shenzen_df = add_image_path(
        TB_shenzen_df, config.TB_SHENZHEN_IMAGES_BASE_PATH)

    # create TB_xray_montgomery dataframe and process it
    TB_montgomery_df = pd.read_csv(config.TB_MONTGOMERY_METADATA_PATH)
    TB_montgomery_df = TB_montgomery_df.rename(
        columns={'study_id': 'Image Index', 'findings': 'Finding Labels'})
    TB_montgomery_df = add_image_path(
        TB_montgomery_df, config.TB_SHENZHEN_IMAGES_BASE_PATH)

    # join the two TB_chest_xrays dataframes
    TB_chest_xrays_df = pd.concat([TB_shenzen_df, TB_montgomery_df])
    TB_chest_xrays_df = TB_chest_xrays_df.loc[:, [
        'Image Index', 'Finding Labels', 'Image Path']]

    # ranaming column names
    TB_chest_xrays_df['Finding Labels'] = TB_chest_xrays_df['Finding Labels'].map(
        lambda finding: 'No Finding' if finding == 'normal' else 'Tuberculosis')

    # change the labels to one hot encoded values
    TB_chest_xrays_df = onehot_encode(TB_chest_xrays_df)

    # keep only some of the columns
    TB_chest_xrays_df = TB_chest_xrays_df.loc[:, config.TB_COLS]

    return TB_chest_xrays_df


def chest_xrays15(chest_xrays14_df, TB_chest_xrays_df):
    # concatinate the two dataframes
    chest_xrays15_df = pd.concat(
        [chest_xrays14_df, TB_chest_xrays_df], ignore_index=True)

    # fill NaN entries with zero.
    chest_xrays15_df = chest_xrays15_df.fillna(0)

    # shuffle the dataset
    chest_xrays15_df = chest_xrays15_df.sample(frac=1)

    return chest_xrays15_df


def train_validation_test_split(df, train_split=.9, val_split=.05):
    # split the dataset into train, validation and test sets
    df = df.sample(frac=1)
    train_end = int(train_split * len(df))
    val_end = int(val_split * len(df)) + train_end
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # define the dataframes that we'll be building
    dataframes = [
        ("training", train_df),
        ("validation", val_df),
        ("testing", test_df)
    ]

    # write the splited dataframes to a file
    for (split_name, df) in dataframes:
        print("[INFO] 'creating {}' split csv file".format(split_name))
        df.to_csv(config.BASE_PATH)

    return (train_df, val_df, test_df)


def move_images(train_df, val_df, test_df):
    # define the datasets that we'll be building
    datasets = [
        ("training", train_df["Image Path"], config.TRAIN_PATH),
        ("validation", val_df["Image Path"], config.VAL_PATH),
        ("testing", test_df["Image Path"], config.TEST_PATH)
    ]

    # loop over the datasets
    for (split_name, image_paths, dst_path) in datasets:
        # show which data split we are creating
        print("[INFO] building '{}' split".format(split_name))

        # if the output base directory does not exist, create it
        if not os.path.exists(dst_path):
            print("[INFO] 'creating {}' directory".format(dst_path))
            os.makedirs(dst_path)

        # loop over the input image paths
        for image_path in image_paths:
            # copy the images to the output path
            try:
                shutil.copy(image_path, dst_path)
            except:
                print("[INFO] image file conflict occured.")


if __name__ == "__main__":
    # ==============================================================
    # chest_xrays14
    chest_xrays14_df = chest_xrays14()
    print("[INFO] chext_xrays14_df count: ", chest_xrays14_df.shape[0])
    print(chest_xrays14_df.sample(3))

    # ==============================================================
    # TB_chest_xrays
    TB_chest_xrays_df = TB_chest_xrays()
    print("[INFO] TB_chest_xrays_df count: ", TB_chest_xrays_df.shape[0])
    print(TB_chest_xrays_df.sample(3))

    # ==============================================================
    # chest_xrays15
    chest_xrays15_df = chest_xrays15(chest_xrays14_df, TB_chest_xrays_df)
    visualize_class_count(chest_xrays15_df, config.CLASS_NAMES)
    print("[INFO] chest_xrays15_df count: ", chest_xrays15_df.shape[0])
    print(chest_xrays15_df.sample(3))
    print(chest_xrays15_df.info())

    # ==============================================================
    # split the dataset into 90% training, 5% validation and 5% test
    (train_df, val_df, test_df) = train_validation_test_split(chest_xrays15_df)

    # move the image from 'chest_xrays/' and 'tuberclusosis/'to train, validtion, and test paths.
    move_images(train_df, val_df, test_df)
