from __future__ import annotations

import numpy as np
import torch

from models.step1_knowledge_topology_encoder import KnowledgeTopologyEncoderV2
from utils.logging_utils import get_logger


log = get_logger("adaptive_tempo.topology")


def train_topology_encoder(prerequisite: np.ndarray, collaborative: np.ndarray,
                           cfg: dict, device: str = "cpu",
                           initial_features: torch.Tensor | None = None):
    graph_cfg = cfg["graph"]
    ablations = cfg.get("ablations", {})
    model = KnowledgeTopologyEncoderV2(
        prerequisite.shape[0], dim=int(graph_cfg["embedding_dim"]),
        n_layers=int(graph_cfg["gcn_layers"]), tau=float(graph_cfg["tau"]),
        initial_features=initial_features).to(device)
    prerequisite_t = torch.as_tensor(prerequisite, dtype=torch.float32, device=device)
    if not ablations.get("use_collaborative_graph", True):
        # A2: Zero out collaborative graph to use only explicit prerequisite view
        collaborative_t = torch.eye(prerequisite.shape[0], dtype=torch.float32, device=device)
    else:
        collaborative_t = torch.as_tensor(collaborative, dtype=torch.float32, device=device)
    model.set_graphs(prerequisite_t, collaborative_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(graph_cfg["lr"]))
    epochs = int(graph_cfg["topology_epochs"])
    for epoch in range(epochs):
        model.train()
        explicit, implicit, _ = model()
        loss = model.contrastive_loss(explicit, implicit)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            log.info("epoch %d/%d contrastive_loss=%.4f", epoch + 1, epochs, loss.item())
    model.eval()
    with torch.no_grad():
        embeddings = tuple(value.detach() for value in model())
    return model, embeddings

