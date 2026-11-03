"""pyKT Deep Knowledge Tracing (DKT) implementation.

Reference:
    https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/dkt.py
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKT(Module):
    def __init__(
        self,
        num_c: int,
        emb_size: int = 64,
        dropout: float = 0.1,
        emb_type: str = "qid",
        emb_path: str = "",
        pretrain_dim: int = 768,
    ):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.hidden_size = int(emb_size)
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        """Forward pass.

        Args:
            q: Concept/question indices tensor of shape (batch_size, seq_len).
            r: Binary responses tensor of shape (batch_size, seq_len) in {0, 1}.
            state: Optional previous recurrent state (h_0, c_0).
            return_state: If True, also return the updated recurrent state.

        Returns:
            y: Predicted mastery probabilities of shape (batch_size, seq_len, num_c).
            next_state: (Optional) Updated LSTM recurrent state (h_n, c_n).
        """
        emb_type = self.emb_type
        if emb_type == "qid":
            if (q < 0).any() or (q >= self.num_c).any():
                raise IndexError(
                    f"DKT received concept index outside [0, {self.num_c - 1}]: "
                    f"min={q.min().item()}, max={q.max().item()}"
                )
            x = q.long() + self.num_c * r.long()
            xemb = self.interaction_emb(x)
        else:
            raise ValueError(f"Unsupported emb_type: {emb_type}")

        h, next_state = self.lstm_layer(xemb, state)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        if return_state or (state is not None):
            return y, next_state
        return y
