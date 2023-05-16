import tensorflow as tf
import numpy as np
import customImputerLayerDefinition as myImputer
from customImputerLayerDefinition import myImputer as myImputer

feature_description = {
    'tickers': tf.io.FixedLenFeature([188], tf.float32, np.zeros(188)),
    'day_of_week': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'month_of_year': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'hour_of_day': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'target': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}

def parse_example(serialized_example):
    example = tf.io.parse_example(serialized_example, feature_description)
    features = {
        'tickers': example['tickers'],
        'day_of_week': example['day_of_week'],
        'month_of_year': example['month_of_year'],
        'hour_of_day': example['hour_of_day'],
    }
    target = example['target']
    return features, target

raw_dataset = tf.data.TFRecordDataset(['dataset.tfrecord'])
datLen = raw_dataset.reduce(0,lambda x,y: x+1)
n_valid = int(datLen.numpy()*.1)
n_test = int(datLen.numpy()*.1)
n_train = datLen.numpy()-n_valid-n_test
train_data = raw_dataset.take(n_train).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

test_data = raw_dataset.skip(n_train).take(n_test).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

valid_data = raw_dataset.skip(n_train+n_test).take(n_valid).batch(2048).map(
    parse_example,num_parallel_calls=8).cache()

inputDict = {
    'tickers': tf.keras.Input(shape=(188,), dtype=tf.float32),
    'day_of_week': tf.keras.Input(shape=(), dtype=tf.int64),
    'month_of_year': tf.keras.Input(shape=(), dtype=tf.int64),
    'hour_of_day': tf.keras.Input(shape=(), dtype=tf.int64)
}

imputer = myImputer()
imputer.adapt(train_data.map(lambda x,y: x['tickers']))
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_data.map(lambda x,y: imputer(x['tickers'])))

day_of_week_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=6,num_oov_indices=0)
day_of_week_catEncoder.adapt(train_data.map(lambda x,y:x['day_of_week']))
day_of_week_catInts=day_of_week_catEncoder(inputDict['day_of_week'])

month_of_year_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=12,num_oov_indices=0)
month_of_year_catEncoder.adapt(train_data.map(lambda x,y:x['month_of_year']))
month_of_year_catInts=month_of_year_catEncoder(inputDict['month_of_year'])

hour_of_day_catEncoder=tf.keras.layers.IntegerLookup(max_tokens=24,num_oov_indices=0)
hour_of_day_catEncoder.adapt(train_data.map(lambda x,y:x['hour_of_day']))
hour_of_day_catInts=hour_of_day_catEncoder(inputDict['hour_of_day'])

day_of_week_embedding = tf.keras.layers.Embedding(6, 2)(day_of_week_catInts)
month_of_year_embedding = tf.keras.layers.Embedding(12, 2)(month_of_year_catInts)
hour_of_day_embedding = tf.keras.layers.Embedding(24, 2)(hour_of_day_catInts)

day_of_week_embedding = tf.keras.layers.Flatten()(day_of_week_embedding)
month_of_year_embedding = tf.keras.layers.Flatten()(month_of_year_embedding)
hour_of_day_embedding = tf.keras.layers.Flatten()(hour_of_day_embedding)

preproced = tf.concat([normalizer(imputer(inputDict['tickers'])), day_of_week_embedding, month_of_year_embedding, hour_of_day_embedding], axis=-1)

restMod = tf.keras.Sequential([
    tf.keras.layers.Dense(1000,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(800,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(600,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')
    ])

decs = restMod(preproced)
whole_model = tf.keras.Model(inputs=inputDict, outputs=decs)
whole_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
whole_model.summary()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('mySavedModel', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

history = whole_model.fit(train_data, epochs=200, verbose=1, validation_data=valid_data, callbacks=[early_stopping_cb, checkpoint_cb])

whole_model.evaluate(test_data)

whole_model.save('mySavedModel')