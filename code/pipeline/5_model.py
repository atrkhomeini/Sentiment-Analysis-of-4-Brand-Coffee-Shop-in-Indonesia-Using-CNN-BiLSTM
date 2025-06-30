# =============================
# 0. Import Library
# =============================
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, GlobalAveragePooling1D, Bidirectional, LSTM, Dropout, Dense, concatenate

# =============================
# 1. Load Dataset & Label Encode
# =============================

df = pd.read_excel('../data/output/labeled_n_aspect.xlsx', sheet_name='Sheet1')
le = LabelEncoder()
df['Encoded_Label'] = le.fit_transform(df['Sentimen'])

# =============================
# 2. Split 80% Train / 20% Holdout
# =============================

df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df['Encoded_Label'], random_state=42
)
# Save train data for validation
df_train.to_excel('../data/output/df_train.xlsx', index=False)
df_test.to_excel('../data/output/df_test.xlsx', index=False)

# after validation data train
df_train = pd.read_excel('../data/data_valid/df_train.xlsx', sheet_name='data_valid')
df_test = pd.read_excel('../data/data_valid/df_test.xlsx', sheet_name='data_test')
le = LabelEncoder()
df_train['Encoded_Label_Valid'] = le.fit_transform(df_train['Validation'])
df_test['Encoded_Label'] = le.transform(df_test['Sentimen'])
# =============================
# 3. Handle Imbalanced Data
# =============================

# for negative validation, augment the data using synonym augmentation
df_neg = df_train[df_train['Validation'] == 'negative'].copy()
aug = naw.SynonymAug(aug_src='wordnet')
df_neg['Augmented_Text'] = df_neg['Text Normalization'].apply(lambda x: aug.augment(x))
df_neg_augmented = df_neg.copy()
df_neg_augmented['Text Normalization'] = df_neg_augmented['Augmented_Text']
df_neg_final = pd.concat([df_neg, df_neg_augmented], ignore_index=True)

df_other = df_train[df_train['Validation'] != 'negative']
df_aug_combined = pd.concat([df_other, df_neg_final], ignore_index=True)

max_count = df_aug_combined['Encoded_Label_Valid'].value_counts().max()
df_balanced = pd.concat([
    resample(sub_df, replace=True, n_samples=max_count, random_state=42)
    for _, sub_df in df_aug_combined.groupby('Encoded_Label_Valid')
], ignore_index=True)

# for aspects, we will do random oversampling 
from sklearn.utils import resample

# Cek distribusi awal
print("Distribusi Awal Aspek:")
print(df_train['Predicted_Aspect'].value_counts())

# Tentukan aspek yang tersedia
aspects = df_train['Predicted_Aspect'].unique()

# Hitung jumlah sampel tertinggi
max_count = df_train['Predicted_Aspect'].value_counts().max()

# Lakukan random oversampling per aspek
balanced_df = pd.DataFrame()

for aspect in aspects:
    subset = df_train[df_train['Predicted_Aspect'] == aspect]
    resampled_subset = resample(subset,
                                replace=True,
                                n_samples=max_count,
                                random_state=42)
    balanced_df = pd.concat([balanced_df, resampled_subset])

# Reset index
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Cek distribusi setelah balancing
print("\nDistribusi Setelah Balancing:")
print(balanced_df['Predicted_Aspect'].value_counts())

# Simpan dataset balanced
balanced_df.to_excel('balanced_aspect_dataset.xlsx', index=False)
print("\nDataset balanced berhasil disimpan sebagai 'balanced_aspect_dataset.xlsx'")


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

X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['Text Normalization']),
                       maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y_test = df_test['Encoded_Label'].values

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


#============================================================================================================
# Use if you dont have the best parameters yet (you can skip this part if you already have the best parameters)
# ===========================================================================================================
'''
# Step 7: Define model builder
from itertools import product
def build_model(cnn_filters, kernel_size, lstm_units, dropout_rate, cnn_activation):
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding = Embedding(input_dim=len(word_index)+1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=True)(input_layer)
    cnn = Conv1D(cnn_filters, kernel_size, activation=cnn_activation)(embedding)
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

from itertools import product

# Step 8. Grid parameter space
param_grid = {
    'cnn_filters': [32, 64, 128],
    'kernel_size': [3, 5, 7],
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'cnn_activation': ['relu', 'sigmoid', 'tanh']
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_score = 0
best_params = None

print("Starting Grid Search...\n")
for params in product(*param_grid.values()):
    cnn_filters, kernel_size, lstm_units, dropout_rate, cnn_activation = params
    print(f"Testing params: filters={cnn_filters}, kernel={kernel_size}, lstm={lstm_units}, dropout={dropout_rate}, cnn_activation={cnn_activation}")
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_model(cnn_filters, kernel_size, lstm_units, dropout_rate, cnn_activation)
        model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)

        y_pred = model.predict(X_val).argmax(axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores)
    print(f"Avg F1: {avg_f1:.4f}\n")

    if avg_f1 > best_score:
        best_score = avg_f1
        best_params = params

print("==== Grid Search Complete ====")
print("Best F1 Score:", best_score)
print("Best Params (filters, kernel, lstm, dropout, cnn_activation):", best_params)
'''
# =============================
# 6. Build Model with Best Param
# =============================
def build_model():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,)) # Hasil Tokenisasi

    embedding = Embedding(input_dim=len(word_index)+1, # Mengubah kata menjadi vector embed berdimensi tetap
                          output_dim=embedding_dim, # outputnya berupa dimensi
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=True)(input_layer) # Embedding diperbolehkan untuk menyesuaikan selama training
    # CNN Layer
    cnn = Conv1D(128, 7, activation='relu')(embedding) # Mendeteksi fitur lokal dengan 7 frasa, mengekstrak 128 jenis fitur, ReLU mengaktifkan sinyal positif
    
    # LSTM Layer
    # input: Hasil CNN yang masih berformat sekuens (3D tensor)
    lstm = Bidirectional(LSTM(64))(cnn) # Memahamai urutan kata 2 arah (maju-mundur). Hidden unit: 64 masing-masing arah
    x = Dropout(0.2)(lstm) # Dropout untuk mengurangi overfitting
    x = Dense(64, activation='relu')(x) # Fully connected layer dengan 64 neuron, ReLU sebagai aktivasi
    output = Dense(3, activation='softmax')(x) # Output layer dengan 3 neuron (jumlah kelas), softmax untuk probabilitas kelas
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# ============================================================================
# 9. 10-Fold Cross Validation on Balanced Train Data
# ============================================================================
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs, precisions, recalls, f1s, losses = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"\nFold {fold}/10")
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

# =============================
# Evaluation for Aspect Classification
# =============================
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Load dataset hasil labeling dan prediksi
df = pd.read_excel('balanced_aspect_dataset.xlsx', sheet_name='Sheet1')  # Pastikan file kamu sesuai

# Ambil kolom yang diperlukan
y_true = df['Valid_Aspect'].astype(str)  # Kolom yang berisi label sebenarnya
y_pred = df['Predicted_Aspect'].astype(str)

# Hitung akurasi keseluruhan
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Hitung precision, recall, f1-score per kelas
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=['Produk', 'Layanan', 'Promosi'])

# Tampilkan hasil per kelas
aspect_labels = ['Produk', 'Layanan', 'Promosi']
for idx, aspect in enumerate(aspect_labels):
    print(f'\nAspect: {aspect}')
    print(f'Precision: {precision[idx]:.4f}')
    print(f'Recall: {recall[idx]:.4f}')
    print(f'F1-Score: {f1[idx]:.4f}')
    print(f'Support: {support[idx]}')

# Tampilkan classification report lengkap
print('\nClassification Report:')
print(classification_report(y_true, y_pred, labels=['Produk', 'Layanan', 'Promosi']))

# Tampilkan confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=['Produk', 'Layanan', 'Promosi'])
print('\nConfusion Matrix:')
print(pd.DataFrame(cm, index=['Produk (True)', 'Layanan (True)', 'Promosi (True)'],
                   columns=['Produk (Pred)', 'Layanan (Pred)', 'Promosi (Pred)']))


# =============================
# 10. Final Training on Full Train Set & Evaluation on Test Set
# =============================

# Train Final Model on Balanced Training Set
model = build_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict on Test Set
y_pred_proba = model.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

# Decode label integer back to original class
y_true_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

from sklearn.metrics import classification_report

print("Classification Report (Test Set):\n")
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))


# =============================
# 11. Visualize Results
# =============================
# 1. Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)

# Plotting
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()

# 2. Bar chart akurasi tiap kelas
import pandas as pd

report = classification_report(y_true_labels, y_pred_labels, target_names=le.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Bar chart F1 per kelas
ax = df_report.iloc[:-3]['f1-score'].plot(kind='bar', color='skyblue')
plt.title("F1-Score per Class on Test Set")
plt.ylabel("F1 Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add data labels above bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# save results
df_test['True Label'] = y_true_labels
df_test['Predicted Label'] = y_pred_labels

df_test[['Text Normalization', 'True Label', 'Predicted Label']].to_csv(
    '../data/output/test_predictions.csv', index=False)

