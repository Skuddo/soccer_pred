import os
import sys

from modules.DataHandler.data_handler import DataHandler


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def loadDataset(path):
    return pd.read_csv(path)

def getMatchResult(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

def predictMatch(home, away):
    input_data = le.transform([[home, away]]).toarray()
    pred = model.predict(input_data)[0]
    print(f"{home} vs {away}")
    print(f"Home win: {pred[0]*100:.2f}%")
    print(f"Draw:     {pred[1]*100:.2f}%")
    print(f"Away win: {pred[2]*100:.2f}%\n")

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 3:
        if sys.argv[1] in ['-d', '--dataset']:
            data_mode = sys.argv[2]

    dh = DataHandler(data_mode)

    # dataset_path = "datasets/worldcup/matches_1930_2022.csv"
    # df = loadDataset(dataset_path)

    # df['result'] = df.apply(getMatchResult, axis=1)

    # le = OneHotEncoder()

    # X = le.fit_transform(df[["home_team","away_team"]]).toarray()
    # Y = df['result']
    # Y_encoded = pd.get_dummies(Y)

    # test_ratio = 0.4

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=test_ratio, random_state=42)

    # model = Sequential([
    #     Input(shape=(X.shape[1],)),
    #     Dense(64, activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(3, activation='softmax'),
    #     ])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(X_train, Y_train, epochs = 20, batch_size=16, validation_split=0.1)

    # predictMatch("Poland", "Brazil")
    # predictMatch("Brazil", "Poland")

    # print(df['result'].value_counts(normalize=True))