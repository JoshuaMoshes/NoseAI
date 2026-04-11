import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .model_abc import Model, test_model


class ScentTransformerNet(nn.Module):
    """
    Simple transformer-based model for classifying scent data. Each sensor reading
    is treated as a token, and we use a transformer encoder to process the sequence
    of sensor readings. The output is mean-pooled and fed into a classifier.

    Helpful resources for this:
    https://github.com/hyunwoongko/transformer
    https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
    https://pureai.substack.com/p/building-a-simple-transformer-using-pytorch
    """
    def __init__(self, num_features, num_classes, d_model, nhead,
                 num_layers, dim_feedforward, dropout):
        """
        num_features: number of sensor readings (tokens)
        num_classes: number of output classes
        d_model: transformer hidden dimension
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dim_feedforward: feedforward dimension in transformer
        dropout: dropout rate in transformer to prevent overfitting
        """
        super().__init__()
        # Each feature is a single scalar, this projects it to d_model dimensions for the transformer
        self.feature_embedding = nn.Linear(1, d_model)

        # Positional encodings help transformer know which token is which,
        # transformers have no inherent sense of order
        # we have 12 features, so we create a learnable positional encoding for each feature position
        # 0.02 is a common small scale for initialization, but this can be tuned
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )

        # TODO: Should be fine to use nn.TransformerEncoderLayer here?
        # TransformerEncoderLayer is a single layer of the transformer, which includes multi-head attention and feedforward network.
        # Similar to a CNN layer or RNN cell, but with attention instead of convolution or recurrence.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # num_layers is how many times encoder layer is stacked.
        # references suggest 2 is common for small datasets like SmellNET
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # layer norm + linear --> classifier logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # Lots of LLM & articles used here to get this and understand it.
        # input x is (batch, num_features)

        # x: (batch, num_features) --> (batch, num_features, 1) for embedding
        x = x.unsqueeze(-1)
        # x: (batch, num_features, 1) --> (batch, num_features, d_model) for transformer
        x = self.feature_embedding(x)
        # add attention positional encoding (learned) to the input features
        x = x + self.pos_encoding
        # this is the main transformer processing step,
        # x: (batch, num_features, d_model) --> (batch, num_features, d_model)
        # every token (feature) computes attention with every other token, and updates its representation
        # output at position i is no longer what sensor i was, but combination of what sensor i read,
        # along with what all other sensors read, filtered through the learned attention pattern
        x = self.transformer_encoder(x)
        # 12 token vectors --> 1 vector by mean pooling
        # there are other options here, but the token count & dataset size are small, so it should be okay
        x = x.mean(dim=1)
        # now x is (batch, d_model) and we can feed it into the classifier to get logits for each class
        x = self.classifier(x)
        # output is (batch, num_classes) with raw logits for each class, which will be used in loss function and prediction
        return x


class TransformerModel(Model):
    """
    Plumbing for training and using the ScentTransformerNet.
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1,
                 lr=1e-3, epochs=30, batch_size=64):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # set during fit
        self.net: ScentTransformerNet | None = None
        self.label_to_index = None
        self.index_to_label: dict[int, str] | None = None
        self.min_vals = None
        self.ranges = None
        self.feature_columns = None

    def _fit_min_max(self, X: np.ndarray):
        """Compute min and range for each feature for min-max normalization."""
        self.min_vals = X.min(axis=0)
        ranges = X.max(axis=0) - self.min_vals
        ranges[ranges == 0] = 1
        self.ranges = ranges

    def _transform_min_max(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max normalization using the fitted min and range values."""
        return (X - self.min_vals) / self.ranges

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode string labels as integers."""
        unique_labels = sorted(set(y))
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        return np.array([self.label_to_index[label] for label in y])

    def _decode_labels(self, indices: np.ndarray) -> np.ndarray:
        if not self.index_to_label:
            raise ValueError("Label decoding failed. No index_to_label mapping found.")
        return np.array([self.index_to_label[i] for i in indices])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._fit_min_max(X)
        X_norm = self._transform_min_max(X)

        y_encoded = self._encode_labels(y)

        if not self.label_to_index or not self.index_to_label:
            raise ValueError("Label encoding failed. Check that labels are non-empty and unique.")

        num_features = X_norm.shape[1]
        num_classes = len(self.label_to_index)

        self.net = ScentTransformerNet(
            num_features=num_features,
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        # dataloader
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        print(f"Training transformer on {self.device}")
        print(f"  features: {num_features}  classes: {num_classes}  "
              f"samples: {len(X_norm)}")
        print(f"  d_model={self.d_model}  heads={self.nhead}  "
              f"layers={self.num_layers}  epochs={self.epochs}")

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.net(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:>3}/{self.epochs} loss: {avg_loss:.4f}  acc: {accuracy:.4f}")

        print("Training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._transform_min_max(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        if not self.net:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        if not self.index_to_label:
            raise ValueError("Label decoding failed. No index_to_label mapping found.")

        self.net.eval()
        with torch.no_grad():
            logits = self.net(X_tensor)
            '''
            probs = torch.softmax(logits, dim=1)

            # print top 3 predictions with confidence
            for i in range(len(X_tensor)):
                top3_probs, top3_indices = probs[i].topk(3)
                top3_labels = [self.index_to_label[idx.item()] for idx in top3_indices]
                print("  top 3:", [(l, f"{p:.3f}") for l, p in zip(top3_labels, top3_probs.tolist())])
            '''

            indices = logits.argmax(dim=1).cpu().numpy()
        return self._decode_labels(indices)

    def save(self, path):
        if not self.net:
            raise ValueError("Model has not been trained yet. Nothing to save.")
        self.net.eval()
        torch.save(self, path)

    @staticmethod
    def load(path):
        model = torch.load(path, weights_only=False)
        model.net.eval()
        return model

if __name__ == "__main__":
    trainingDataDataFrame = pd.read_csv("training-data.csv")
    testingDataDataFrame = pd.read_csv("testing-data.csv")

    TARGET_COLUMN = "type"

    model = TransformerModel(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        lr=1e-3,
        epochs=15,
        batch_size=64,
    )

    model.fit(
        trainingDataDataFrame.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float),
        trainingDataDataFrame[TARGET_COLUMN].to_numpy()
    )

    test_model(
        model,
        testingDataDataFrame.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float),
        testingDataDataFrame[TARGET_COLUMN].to_numpy()
    )

    sample_values = [54, 94, 130, 765, 2, 12, 0, 33.59, 688.6, 100.0, 0.0, 3170.54]
    sample_values = np.array(sample_values, dtype=np.float32).reshape(1, -1)
    print("Predicted:", model.predict(sample_values))

    model.save("transformer.pt")
