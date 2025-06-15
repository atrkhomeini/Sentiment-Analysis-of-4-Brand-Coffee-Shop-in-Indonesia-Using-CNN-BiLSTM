# =============================
# 0. Import Library
# =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalAveragePooling1D, Bidirectional, LSTM, Dropout, Dense, concatenate

# =============================
# 1. Load Dataset & Label Encode
# =============================
df = pd.read_csv('../data/output/indobert_labeled_data.csv')
le = LabelEncoder()
df['Encoded_Label'] = le.fit_transform(df['Label_Bert'])

# =============================
# 2. Split 80% Train / 20% Holdout
# =============================
df_train, df_holdout = train_test_split(
    df, test_size=0.2, stratify=df['Encoded_Label'], random_state=42
)
# Save train data for validation
df_train.to_csv('../data/output/indobert_train.csv', index=False)

# after validation data train
df_train = pd.read_excel('../data/data_valid/main_indobert_train.xlsx', sheet_name='data_valid')
le = LabelEncoder()
df_train['Encoded_Label_Valid'] = le.fit_transform(df_train['Validation'])
# =============================
# 3. Handle Imbalanced Data
# =============================
max_count = df_train['Encoded_Label_Valid'].value_counts().max()
df_balanced = pd.concat([
    resample(sub_df, replace=True, n_samples=max_count, random_state=42)
    for _, sub_df in df_train.groupby('Encoded_Label_Valid')
], ignore_index=True)

# =============================
# 4. Tokenization
# =============================
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 50
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_balanced['Text Normalization'])

X_train = pad_sequences(tokenizer.texts_to_sequences(df_balanced['Text Normalization']),
                        maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y_train = df_balanced['Encoded_Label'].values

X_holdout = pad_sequences(tokenizer.texts_to_sequences(df_holdout['Text Normalization']),
                          maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y_holdout = df_holdout['Encoded_Label'].values

# =============================
# 5. Load GloVe Embedding
# =============================
embedding_index = {}
with open("../src/glove/glove.6B.300d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

embedding_dim = 300
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    vector = embedding_index.get(word)
    if vector is not None:
        embedding_matrix[i] = vector

# =============================
# 6. Build Model with Best Param
# =============================
def build_model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding = Embedding(input_dim=len(word_index)+1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=True)(input_layer)
    cnn = Conv1D(64, 3, activation='relu')(embedding)
    cnn = GlobalAveragePooling1D()(cnn)
    lstm = Bidirectional(LSTM(64))(embedding)
    x = concatenate([cnn, lstm])
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

'''
# Step 6: Define model builder
def build_model(cnn_filters, kernel_size, lstm_units, dropout_rate):
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding = Embedding(input_dim=len(word_index)+1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=True)(input_layer)
    cnn = Conv1D(cnn_filters, kernel_size, activation='relu')(embedding)
    cnn = GlobalAveragePooling1D()(cnn)
    lstm = Bidirectional(LSTM(lstm_units))(embedding)
    x = concatenate([cnn, lstm])
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Step 7: Grid search
param_grid = {
    'cnn_filters': [64, 128],
    'kernel_size': [3, 5],
    'lstm_units': [32, 64],
    'dropout_rate': [0.3, 0.5]
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_params = None

for params in product(*param_grid.values()):
    cnn_filters, kernel_size, lstm_units, dropout_rate = params
    print(f"\nTesting config: {params}")
    f1s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        model = build_model(cnn_filters, kernel_size, lstm_units, dropout_rate)
        model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(X_val).argmax(axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1s.append(f1)

    avg_f1 = np.mean(f1s)
    print(f"Avg F1: {avg_f1:.4f}")

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params

best_score, best_params
'''

# =============================
# 9. 5-Fold Cross Validation on Balanced Train Data
# =============================
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, precisions, recalls, f1s, losses = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\nFold {fold}/5")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = build_model()
    model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)

    y_pred_proba = model.predict(X_val)
    y_pred = y_pred_proba.argmax(axis=1)

    accs.append(accuracy_score(y_val, y_pred))
    precisions.append(precision_score(y_val, y_pred, average='macro'))
    recalls.append(recall_score(y_val, y_pred, average='macro'))
    f1s.append(f1_score(y_val, y_pred, average='macro'))
    losses.append(log_loss(y_val, y_pred_proba))

# Summary
print("\n===== 5-Fold CV Results on Balanced Training Data =====")
print("Average Accuracy: {:.2f}%".format(np.mean(accs) * 100))
print("Average Precision: {:.2f}%".format(np.mean(precisions) * 100))
print("Average Recall: {:.2f}%".format(np.mean(recalls) * 100))
print("Average F1-Score: {:.2f}%".format(np.mean(f1s) * 100))
print("Average Log Loss: {:.2f}".format(np.mean(losses)))

