# train_lstm.py
from models.lstm import LSTMModel

model = LSTMModel(window=10, stride=3, epochs=100)
model.fit_from_csv("collected-data.csv", ckpt_path="models/lstm.pt")
