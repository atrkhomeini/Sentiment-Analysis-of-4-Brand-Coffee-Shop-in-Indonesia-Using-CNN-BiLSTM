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