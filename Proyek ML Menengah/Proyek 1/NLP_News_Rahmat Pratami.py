from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

berita1 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-1.csv'
berita2 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-2.csv'
berita3 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-3.csv'
berita4 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-4.csv'
berita5 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-5.csv'
berita6 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-6.csv'
berita7 = '/content/drive/MyDrive/Datasets/News Classification/inshort_news_data-7.csv'

df1 = pd.read_csv(berita1)
df2 = pd.read_csv(berita2)
df3 = pd.read_csv(berita3)
df4 = pd.read_csv(berita4)
df5 = pd.read_csv(berita5)
df6 = pd.read_csv(berita6)
df7 = pd.read_csv(berita7)

df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0, ignore_index=True)
df.head()

df.shape

df_baru = df.drop(columns=['news_headline','Unnamed: 0'])
df_baru

import pandas as pd
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('all')
# Kecilkan huruf
df_baru['news_article'] = df_baru['news_article'].apply(lambda x: x.lower())

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()

def lem(data):
    pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
    return ' '.join([lemmatizer.lemmatize(w, pos_dict.get(t, wn.NOUN)) for w, t in nltk.pos_tag(data.split())])

df_baru['news_article'] = df_baru['news_article'].apply(lambda x: lem(x))

# Hapus tanda baca
def cleaner(data):
    return data.translate(str.maketrans('', '', string.punctuation))

df_baru['news_article'] = df_baru['news_article'].apply(lambda x: cleaner(x))

# Hapus nomor
def rem_numbers(data):
    return re.sub('[0-9]+', '', data)

df_baru['news_article'] = df_baru['news_article'].apply(rem_numbers)

# Menghapus stopwords
st_words = set(stopwords.words('english'))  # Assuming English stopwords
def stopword(data):
    return ' '.join([w for w in data.split() if w not in st_words])

df_baru['news_article'] = df_baru['news_article'].apply(lambda x: stopword(x))

df_baru.head()

news_category = pd.get_dummies(df_baru.news_category)
df_baru1 = pd.concat([df_baru, news_category], axis=1)
df_baru1 = df_baru1.drop(columns='news_category')
df_baru1

print(df_baru1.columns.tolist())

judul = df_baru1['news_article'].values
label = df_baru1.drop('news_article', axis=1).values

judul

label

from sklearn.model_selection import train_test_split
judul_train, judul_val, label_train, label_val = train_test_split(judul, label, test_size=0.2, random_state=123)

max_len = 256

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(lower=True, char_level=False)
tokenizer.fit_on_texts(judul_train)
tokenizer.fit_on_texts(judul_val)

sequences_train = tokenizer.texts_to_sequences(judul_train)
sequences_val = tokenizer.texts_to_sequences(judul_val)

padded_train = pad_sequences(sequences_train, maxlen=max_len)
padded_val = pad_sequences(sequences_val, maxlen=max_len)

word_to_index = tokenizer.word_index

vocab_size =  len(word_to_index)
oov_tok = "<OOV>"
embedding_dim = 100

import numpy as np
embeddings_index = {};

with open('/content/drive/MyDrive/Datasets/Video Game Ratings and Reviews Dataset/glove/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

# Ensure that vocab_size matches the expected size
embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))

for word, i in word_to_index.items():
    if i <= vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embeddings_matrix], trainable=False, input_length=max_len),
    tf.keras.layers.LSTM(256, return_sequences=False),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

class callback_acc(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):
      self.model.stop_training = True

callback_acc = callback_acc()

num_epochs = 50
history = model.fit(padded_train,
                    label_train,
                    epochs=num_epochs,
                    batch_size=32,
                    callbacks=[callback_acc],
                    validation_data=(padded_val, label_val), verbose=1)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Nama : Rahmat Pratami
# Email : exzaardyansyah894@gmail.com