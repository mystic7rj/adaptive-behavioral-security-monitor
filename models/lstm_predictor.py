from __future__ import annotations

import logging

from torch import Tensor, nn

logger = logging.getLogger(__name__)


class SessionLSTM(nn.Module):
    """Model event-token sequences and score sequence plausibility via log-likelihood.

    Sequence modeling complements vector anomalies by capturing temporal ordering behavior.
    """

    def __init__(self, vocab_size: int = 32, embedding_dim: int = 16, hidden_size: int = 64) -> None:
        """Initialize embedding + stacked LSTM network for next-token prediction."""

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, seq: Tensor) -> Tensor:
        """Compute token log-probabilities for each timestep in the input sequence."""

        embedded = self.embedding(seq)
        outputs, _ = self.lstm(embedded)
        logits = self.projection(outputs)
        output: Tensor = self.log_softmax(logits)
        return output

    def sequence_log_likelihood(self, seq: Tensor) -> float:
        """Return summed next-token log-likelihood used as sequence normality score."""

        if seq.dim() == 1:
            seq = seq.unsqueeze(0)  # Promote single sequence input to batch shape.
        if seq.size(1) < 2:
            raise ValueError("sequence length must be at least 2")

        inputs = seq[:, :-1]
        targets = seq[:, 1:]
        log_probs = self.forward(inputs)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())
