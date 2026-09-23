"""PyTorch LSTM forecaster for univariate Favorita sales series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


class _SalesLSTM(nn.Module):
    """Minimal recurrent network predicting the next scaled sales value."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return the final-step regression output."""
        encoded, _ = self.lstm(sequence)
        return self.output(encoded[:, -1, :])


class LSTMForecaster:
    """Recursive univariate LSTM implementing the project forecast model contract."""

    name = "lstm"

    def __init__(
        self,
        lookback: int = 28,
        hidden_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.001,
        target_column: str = "sales",
        random_state: int = 42,
    ) -> None:
        if min(lookback, hidden_size, epochs) < 1 or learning_rate <= 0:
            raise ValueError("Lookback, hidden size, epochs, and learning rate must be positive")
        self.lookback, self.hidden_size, self.epochs = lookback, hidden_size, epochs
        self.learning_rate, self.target_column, self.random_state = learning_rate, target_column, random_state
        self._network: _SalesLSTM | None = None
        self._mean = 0.0
        self._std = 1.0
        self._history: list[float] = []

    def fit(self, train: pd.DataFrame) -> "LSTMForecaster":
        """Fit on overlapping lookback sequences from one chronologically ordered series."""
        self._validate_frame(train)
        values = train.sort_values("date")[self.target_column].astype(float).to_numpy()
        if len(values) <= self.lookback:
            raise ValueError("Training data must contain more rows than the LSTM lookback")
        self._mean = float(values.mean())
        self._std = float(values.std()) or 1.0
        scaled = (values - self._mean) / self._std
        inputs = np.array([scaled[offset : offset + self.lookback] for offset in range(len(scaled) - self.lookback)])
        labels = scaled[self.lookback :]
        torch.manual_seed(self.random_state)
        self._network = _SalesLSTM(self.hidden_size)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        features = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
        targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        self._network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self._network(features), targets)
            loss.backward()
            optimizer.step()
        self._history = values.tolist()
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Forecast recursively, feeding predictions—not future actuals—into each next window."""
        if self._network is None:
            raise RuntimeError("Call fit before predict")
        self._validate_frame(future, require_target=False)
        history = self._history.copy()
        predictions: list[float] = []
        self._network.eval()
        with torch.no_grad():
            for _ in future.sort_values("date").itertuples():
                context = (np.asarray(history[-self.lookback:]) - self._mean) / self._std
                tensor = torch.tensor(context, dtype=torch.float32).reshape(1, self.lookback, 1)
                prediction = max(0.0, float(self._network(tensor).item() * self._std + self._mean))
                history.append(prediction)
                predictions.append(prediction)
        ordered_index = future.sort_values("date").index
        return pd.Series(predictions, index=ordered_index, name="prediction").reindex(future.index)

    def _validate_frame(self, frame: pd.DataFrame, require_target: bool = True) -> None:
        """Ensure the frame describes exactly one store and product family."""
        required = {"date", "store_nbr", "family"}
        if require_target:
            required.add(self.target_column)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"LSTM frame missing columns: {sorted(missing)}")
        if len(frame[["store_nbr", "family"]].drop_duplicates()) != 1:
            raise ValueError("LSTMForecaster accepts one store/family series per fit")
