import pandas as pd

df = pd.read_csv('../data/output/indobert_labeled_data.csv')

# =============================
# 1. Tokenizing
# =============================
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 50

# Tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
X_all = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Label encoding
le = LabelEncoder()
y_all = le.fit_transform(df['Label_Bert'])

# =============================
# 2. Load GloVe & Embedding Matrix
# =============================
import numpy as np

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
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# =============================
# 3. Class Weighting
# =============================
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_all),
                                     y=y_all)
class_weight_dict = dict(enumerate(class_weights))

# =============================
# 4. Split Data: 90% train-val, 10% test
# =============================
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all, test_size=0.1, stratify=y_all, random_state=42)

# =============================
# 5. Build CNN-BiLSTM Function
# =============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2

def build_model(cnn_filters, kernel_size, lstm_units, dropout_rate):
    model = Sequential([
        Embedding(input_dim=len(word_index)+1,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=True),
        Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(lstm_units)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# =============================
# 6. K-Fold Cross Validation
# =============================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accs, precisions, recalls, f1s, losses = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val, y_train_val), 1):
    print(f"\nFold {fold}/10")
    X_tr, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_tr, y_val = y_train_val[train_idx], y_train_val[val_idx]

    model = build_model(128,5,64,0.5)
    model.fit(X_tr, y_tr,
              epochs=10,
              batch_size=32,
              class_weight=class_weight_dict,
              verbose=0)

    y_pred_proba = model.predict(X_val)
    y_pred = y_pred_proba.argmax(axis=1)

    accs.append(accuracy_score(y_val, y_pred))
    precisions.append(precision_score(y_val, y_pred, average='macro'))
    recalls.append(recall_score(y_val, y_pred, average='macro'))
    f1s.append(f1_score(y_val, y_pred, average='macro'))
    losses.append(log_loss(y_val, y_pred_proba))

# =============================
# 7. Evaluation Summary
# =============================
print("\n===== K-Fold Cross Validation Results =====")
print("Average Accuracy: {:.2f}%".format(np.mean(accs) * 100))
print("Average Precision: {:.2f}%".format(np.mean(precisions) * 100))
print("Average Recall: {:.2f}%".format(np.mean(recalls) * 100))
print("Average F1-Score: {:.2f}%".format(np.mean(f1s) * 100))
print("Average Log Loss: {:.2f}".format(np.mean(losses)))


#=============================
# 8. Analyze Errors
#=============================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Ubah label angka ke label asli
y_true_labels = le.inverse_transform(y_val)
y_pred_labels = le.inverse_transform(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Fold Terakhir")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# Create a DataFrame to compare true labels with predicted labels
comparison_df = pd.DataFrame({
    'True Label': y_true_labels,
    'Predicted Label': y_pred_labels
})

# Display the first few rows of the DataFrame
print(comparison_df.head())

comparison_df