from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

df = pd.read_csv('../data/output/4_class_weights.csv')
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
