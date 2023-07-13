# src/dataset/data.py

import json
import pandas as pd
from pathlib import Path


def read_df(image_folder, data_file, train=False):
    """Read dataset dataframe from image folder and json file.

    Args:
        image_folder (str): Folder path containing images.
        data_file (str): Path to json file containing dataset info.
        train (bool, optional): Whether this is for training set. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing image ids, filenames, questions, answers etc.
    """

    # Load data json
    with open(data_file) as f:
        data = json.load(f)

    # Convert images to dataframe
    img_df = pd.DataFrame.from_records(data["images"])

    # Convert annotations to dataframe
    ann_df = pd.DataFrame.from_records(data["annotations"])

    # Merge image infos with annotations
    img_df.rename(columns={"id": "image_id"}, inplace=True)
    img_df["image"] = img_df["filename"].apply(lambda x: f"{image_folder}/{x}")
    df = pd.merge(img_df, ann_df, on="image_id")

    # Preprocess if training set
    if train:
        # Preprocess training df
        df = preprocess_train_df(df)

    return df


def preprocess_train_df(df):
    """Preprocess training dataframe by filtering and modifying fields.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """

    # Preprocess steps
    df = filter_blacklist_ids(df)
    df["answer"] = replace_answer_words(df)

    return df


# Helper functions
def filter_blacklist_ids(df):
    # Filter blacklist image ids
    black_ids = [
        1493,
        2397,
        2900,
        2913,
        2952,
        2955,
        2956,
        2959,
        2989,
        4094,
        10008,
        10009,
        10012,
        10082,
        10083,
        10013,
        10014,
        10015,
        10088,
        10018,
        10089,
        10091,
        10022,
        10092,
        10093,
        10023,
        10024,
        10097,
        10028,
        10098,
        10099,
        10101,
        10030,
        10031,
        10033,
        10104,
        10034,
        10035,
        10108,
        10109,
        10110,
        10111,
        10112,
        10113,
        10114,
        10115,
        10044,
        10048,
        10117,
        10118,
        10119,
        10050,
        10051,
        10121,
        10052,
        10125,
        10054,
        10056,
        10127,
        10129,
        10130,
        10131,
        10132,
        10133,
        10134,
        10064,
        10065,
        10135,
        10138,
        10068,
        10069,
        10139,
        10071,
        10142,
        10144,
        10145,
        10146,
        10075,
        10148,
        10149,
        10150,
        10151,
        18584,
        20381,
        21394,
        1119,
        16850,
        22091,
        12154,
        23370,
        23516,
    ]
    return df[~df["id"].isin(black_ids)]


def replace_answer_words(df):
    # Replace number words in answers
    return df.apply(lambda x: replace_words(x), axis=1)


def replace_word(lang, length, text):
    if (lang == "ja") & (length == 1):
        dicts = {"2": "二", "6": "六", "7": "七"}
        for key, value in dicts.items():
            text = text.replace(value, key)
        return text
    elif (lang == "vi") & (length == 1):
        dicts = {
            "hai": "2",
            "ba": "3",
        }
        for key, value in dicts.items():
            text = text.replace(value, key)
        return text
    else:
        return text
