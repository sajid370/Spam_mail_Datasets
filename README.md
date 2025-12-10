Spam Email Detection Using BERT Embeddings & Multi-Window CNN

A deep learning framework combining BERT contextual embeddings with a multi-window Convolutional Neural Network (CNN) to classify spam vs. non-spam emails with high accuracy.

This project achieves:

Accuracy: 98.69%

AUC: 0.9981

F1-Score: 0.9724

MCC: 0.9639

📌 Dataset

The dataset used consists of 5,728 labeled emails, including:

Spam: 1,368

Ham (Non-Spam): 4,360

📥 Download Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

🧠 Model Architecture
1. BERT Embedding Generation

Model: BERT-base (uncased)

Produces a sequence of contextual vectors, each of size 768

Input length: up to 512 tokens

2. Multi-Window CNN

The CNN extracts discriminative patterns using window sizes:

2, 4, 6

Steps include:

1D Convolution

ReLU activation

Max-Pooling

Feature concatenation

3. Fully Connected Layer + Softmax

A dense layer with 128 units followed by a Softmax classifier outputs spam/ham probabilities.

📊 Results (Summary)
Metric	Value
Accuracy	98.69%
AUC	0.9981
F1 Score	0.9724
MCC	0.9639
Precision	0.9814
Recall	0.9635
📂 Project Structure
├── data/
│   ├── spam.csv
│   ├── ham.csv
│   └── merged_dataset.csv
│
├── models/
│   ├── bert_cnn_model.h5
│   └── tokenizer/
│
├── src/
│   ├── preprocess.py
│   ├── bert_embeddings.py
│   ├── cnn_classifier.py
│   ├── train.py
│   └── evaluate.py
│
├── notebook/
│   └── experiment.ipynb
│
└── README.md
