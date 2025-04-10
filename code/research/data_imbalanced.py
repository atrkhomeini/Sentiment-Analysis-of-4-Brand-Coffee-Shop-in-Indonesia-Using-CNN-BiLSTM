#-----------------------------------------------------------------------------
# Data Imbalanced
#-----------------------------------------------------------------------------
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

df = pd.read_csv('../data/output/3_labeling.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['Sentiment'])

# Cek mapping label
print(dict(zip(le.classes_, le.transform(le.classes_))))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label_encoded']),
    y=df['label_encoded']
)

# Konversi ke dict
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

#-----------------------------------------------------------------------------
# Data Tokenization
#-----------------------------------------------------------------------------

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameter
MAX_NUM_WORDS = 10000   # jumlah maksimal vocab
MAX_SEQUENCE_LENGTH = 50  # panjang maksimal tweet

# Inisialisasi tokenizer dan fit ke teks
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Normalized_Text_Slang'])

# Konversi teks jadi token numerik
sequences = tokenizer.texts_to_sequences(df['Normalized_Text_Slang'])

# Padding agar semua sequence punya panjang sama
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Label target
y = df['label_encoded'].values

#------------------------------------------------------------------------------
# Glove Embedding & Build Model CNN-BiLSTM
#------------------------------------------------------------------------------

#Glove Embedding
import numpy as np

# Load GloVe dari file txt (misal: 100d Bahasa Indonesia)
embedding_index = {}
with open("../src/glove/glove_50dim_wiki.id.case.text.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Buat embedding matrix
word_index = tokenizer.word_index
embedding_dim = 50  # contoh: glove.6B.100d.txt
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Model CNN-BiLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=len(word_index)+1,
              output_dim=embedding_dim,
              weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH,
              trainable=False),  # freeze GloVe weights
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # multiclass: positive, negative, neutral
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#------------------------------------------------------------------------------
# Train Model
#------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=10,
          batch_size=32,
          class_weight=class_weight_dict)
