import tensorflow as tf
from tensorflow.train import FloatList,Int64List, Feature,Features,Example
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request

data_path = Path('datasets/appml-assignment1-dataset-v2.pkl')
if not data_path.is_file():
  Path('datasets').mkdir(parents=True, exist_ok=True)
  url = "https://github.com/HemangiP17/appml-assignment1/blob/main/appml-assignment1-dataset-v2.pkl?raw=true"
  urllib.request.urlretrieve(url, filename=data_path)
pkl = pd.read_pickle("./datasets/appml-assignment1-dataset-v2.pkl")
X = pkl['X']
y = pkl['y']

change = (X['CAD-high'] - X['CAD-close']) / X['CAD-close']
bins = np.linspace(-0.001, 0.001, 21)
target = np.digitize(change, bins=bins)

dates = pd.to_datetime(X['date'])
day_of_week = dates.dt.day_of_week.astype('category')
hour_of_day = dates.dt.hour.astype('category')
month_of_year = dates.dt.month.astype('category')
tickers = X.iloc[:, 1:]


with tf.io.TFRecordWriter('dataset.tfrecord') as f:
    for index in range(len(X)):
        feature={
            'tickers':Feature(float_list=FloatList(value=tickers.values[index])),
            'day_of_week':Feature(int64_list=Int64List(value=[day_of_week.values[index]])),
            'month_of_year':Feature(int64_list=Int64List(value=[month_of_year.values[index]])),
            'hour_of_day':Feature(int64_list=Int64List(value=[hour_of_day.values[index]])),
            'target':Feature(int64_list=Int64List(value=[target[index]])),
        }
        myExamp=Example(features=Features(feature=feature))
        f.write(myExamp.SerializeToString())