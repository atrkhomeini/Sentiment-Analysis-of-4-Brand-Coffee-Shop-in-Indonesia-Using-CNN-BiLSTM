import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout
from keras.initializers import Constant

# ===========================
# DATABASE FETCH
# ===========================
import mysql.connector
connect = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Time12:30',
    database='tugas_akhir'
)
cursor = connect.cursor()
query = "SELECT dates, texts FROM kopi"
cursor.execute(query)
df = pd.DataFrame(cursor.fetchall(), columns=['Date','Text'])
cursor.close()
connect.close()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  
df['Date'] = df['Date'].dt.strftime('%m/%Y')

# ===========================
# TEXT CLEANING + TOKENIZATION
# ===========================
def read_file_as_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return dict(line.strip().split(':', 1) for line in f if ':' in line)

root_dict = read_file_as_dict('../src/NLP_bahasa_resources/combined_root_words.txt')
slang_dict = read_file_as_dict('../src/NLP_bahasa_resources/combined_slang_words.txt')
stopwords_dict = set(open('../src/NLP_bahasa_resources/combined_stop_words.txt', 'r', encoding='utf-8').read().splitlines())
factory = StopWordRemoverFactory()
indonesian_stopwords = set(factory.get_stop_words())

def normalize_text(text, slang_dict, root_dict, stopwords_dict):
    tokens = word_tokenize(text.lower())
    normalized = []

    for token in tokens:
        # Convert slang
        if token in slang_dict:
            token = slang_dict[token]

        # Convert to root/base form
        if token in root_dict:
            token = root_dict[token]

        # Remove stopwords
        if token in stopwords_dict:
            continue

        normalized.append(token)

    return normalized

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)                      # remove URLs
    text = re.sub(r'@\w+', '', text)                         # remove mentions
    text = re.sub(r'#\w+', '', text)                         # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)                      # remove punctuation
    text = re.sub(r'\d+', '', text)                          # remove numbers
    return text.strip()

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in indonesian_stopwords]
    return tokens

def preprocess_dataframe(df, text_col='Text'):
    df = df.copy()
    df['cleaned'] = df[text_col].apply(clean_text)
    df['tokens'] = df['cleaned'].apply(lambda x: normalize_text(x, slang_dict, root_dict, stopwords_dict))
    return df


df = preprocess_dataframe(df)

# ===========================
# TOKENIZER + PADDING
# ===========================
texts = df['tokens'].apply(lambda tokens: ' '.join(tokens)).tolist()
max_len = 50
num_words = 10000

tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# ===========================
# GLOVE EMBEDDING LOADER
# ===========================
def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Provide your actual GloVe file path
glove_path = '../src/glove/glove.6B.100d.txt'
embedding_matrix = load_glove_embeddings(glove_path, tokenizer.word_index, 100)

# ===========================
# CNN + BiLSTM MODEL
# ===========================
def build_cnn_bilstm_model(vocab_size, embedding_matrix, max_len, embedding_dim=100, num_classes=3):
    input_ = Input(shape=(max_len,))
    
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_len,
        trainable=False
    )(input_)

    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_, outputs=output)
    return model

model = build_cnn_bilstm_model(
    vocab_size=len(tokenizer.word_index)+1,
    embedding_matrix=embedding_matrix,
    max_len=max_len,
    embedding_dim=100,
    num_classes=3  # Positive, Negative, Neutral
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ======= Placeholder Info =======
# Your data still needs a `Sentiment` column with 3-class labeling.
# After labeling, convert y to one-hot using:
# from keras.utils import to_categorical
# y = to_categorical(df['label'].values)

#===============================
# IF I DO NOT HAVE A LABELING DATASET
#===============================
from textblob import TextBlob
from googletrans import Translator

translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, src='id', dest='en')
        return translated.text
    except:
        return text  # fallback if translation fails

# Translate cleaned tweet to English
df['translated'] = df['cleaned'].apply(translate_to_english)

# Classify sentiment using TextBlob
def classify_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['translated'].apply(classify_sentiment)

# Show label distribution
print(df['Sentiment'].value_counts())


#================================
# EVALUATE MODEL
#================================
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])
y = to_categorical(df['label'].values, num_classes=3)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights_dict = dict(enumerate(class_weights))

# K-Fold Setup
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1
metrics_summary = []

for train_idx, val_idx in kfold.split(padded_sequences, df['label']):
    print(f"\n📦 Fold {fold_no} Starting...")

    X_train, X_val = padded_sequences[train_idx], padded_sequences[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    y_val_labels = df['label'].values[val_idx]  # for classification report

    # Build model
    model = build_cnn_bilstm_model(
        vocab_size=len(tokenizer.word_index)+1,
        embedding_matrix=embedding_matrix,
        max_len=max_len,
        embedding_dim=100,
        num_classes=3
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[es],
        verbose=1
    )

    # Evaluation
    y_pred_probs = model.predict(X_val)
    y_pred = y_pred_probs.argmax(axis=1)

    acc = np.mean(y_pred == y_val_labels)
    logloss = log_loss(y_val_labels, y_pred_probs)

    print(f"✅ Fold {fold_no} Accuracy: {acc:.4f}")
    print(f"🧮 Fold {fold_no} Log Loss: {logloss:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_val_labels, y_pred, target_names=label_encoder.classes_))

    # Save metrics summary
    metrics_summary.append({
        'fold': fold_no,
        'accuracy': acc,
        'log_loss': logloss
    })

    # Confusion Matrix
    cm = confusion_matrix(y_val_labels, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - Fold {fold_no}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    fold_no += 1

# === Final Summary ===
df_metrics = pd.DataFrame(metrics_summary)
print("\n📊 Final Evaluation Summary:")
print(df_metrics.describe()[['accuracy', 'log_loss']])

