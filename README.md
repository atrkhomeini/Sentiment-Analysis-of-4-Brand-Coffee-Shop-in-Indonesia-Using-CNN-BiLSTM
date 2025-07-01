# 🌟 Sentiment Analysis of Four Coffee Shop Brands in Indonesia Using CNN-BiLSTM

## 📄 Project Overview

This project delivers a comprehensive **sentiment analysis system** targeting customer opinions about four well-known coffee shop brands in Indonesia: **Fore Coffee, Kopi Kenangan, Point Coffee, and Tomoro Coffee**. Leveraging a deep learning model that combines **Convolutional Neural Networks (CNN)** with **Bidirectional Long Short-Term Memory (BiLSTM)** networks, the system classifies tweets into three sentiment categories: positive, negative, and neutral.

## 🤖 Key Features

- Twitter data collection based on brand-specific keywords
- Text preprocessing: normalization, stemming, stopword removal
- Word embedding using pre-trained **GloVe vectors**
- Hybrid CNN-BiLSTM deep learning architecture for sentiment classification
- Evaluation using **10-fold cross-validation** for robustness
- Visualization of performance metrics and sentiment word clouds

## 🔧 Technologies Used

- **Python**: NumPy, Pandas, Scikit-learn
- **TensorFlow / Keras** for deep learning modeling
- **NLTK, Sastrawi, Spacy, TextBlob** for NLP preprocessing
- **Matplotlib, Seaborn, WordCloud** for data visualization

## 📁 Project Structure

```
.
├── data/                  # Datasets for the project
│   ├── raw/               # Raw, immutable data
│   └── output/            # Processed data
├── models/                # Trained and serialized models
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── pipeline/          # Data processing and modeling pipeline
│   ├── scripts/           # Helper scripts
│   └── main.py            # Main script to run the pipeline
├── .gitignore             # Files and folders ignored by Git
├── README.md              # Project description
```

## 🗃️ Dataset

- Tweets were scraped from Twitter using relevant keywords
- Brands: `fore`, `kopi kenangan`, `point coffee`, and `tomoro`
- All data was preprocessed and manually labeled into sentiment categories

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/atrkhomeini/Sentiment-Analysis-of-4-Brand-Coffee-Shop-in-Indonesia-Using-CNN-BiLSTM.git
cd Sentiment-Analysis-of-4-Brand-Coffee-Shop-in-Indonesia-Using-CNN-BiLSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python src/main.py
```

## 🧠 Model Architecture

- **CNN** layers extract local patterns and semantic cues (n-gram features)
- **BiLSTM** captures contextual dependencies from both forward and backward directions
- **Dense layer with softmax** classifies the output into sentiment categories

## 📈 Results Summary

- Achieved over 90% accuracy on validation data
- Fine-tuned using optimal CNN filter sizes, embedding dimensions, and dropout rates
- Output visualizations include accuracy/loss curves, confusion matrix, and sentiment word clouds

## 👤 Author

**Ayat Tulloh Rahulloh Khomeini**  
Undergraduate in Information Systems, Institut Teknologi Sepuluh Nopember (ITS)  
📫 Connect via [LinkedIn](www.linkedin.com/in/ayat-tulloh-rk) or [GitHub](https://github.com/atrkhomeini)

## 📝 Citation

If you find this project useful in your research or development, please cite it as:

```text
@misc{atrkhomeini2025sentiment,
  author       = {Ayat Tulloh Rahulloh Khomeini},
  title        = {Sentiment Analysis of Four Coffee Shop Brands in Indonesia Using CNN-BiLSTM},
  year         = {2025},
  url          = {https://github.com/atrkhomeini/Sentiment-Analysis-of-4-Brand-Coffee-Shop-in-Indonesia-Using-CNN-BiLSTM}
}
```

## ⚠️ Disclaimer

This repository is intended for **educational and research purposes only**. All brand names and trademarks are the property of their respective owners.

---

Built with ❤️ by Ayat Tulloh Rahulloh Khomeini
