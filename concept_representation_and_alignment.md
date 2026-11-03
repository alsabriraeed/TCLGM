# Step 1: Concept Representation Modeling and Multi-View Alignment

This document provides the mathematically verified description and code verification of the dual-view concept representation and alignment module (Step 1: Knowledge Topology Encoder), aligned directly with the implementation in [`models/step1_knowledge_topology_encoder.py`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py) and [`utils/graph_utils.py`](file:///home/raed/TCLGM/utils/graph_utils.py).

---

## 1. Mathematical Formulation (Preview-Ready Markdown)

For modeling concept representations, let $X_0 \in \mathbb{R}^{K \times d_0}$ contain the initial features of the $K$ concepts (e.g., character $n$-gram representations). The initial representations are obtained via a linear projection with Layer Normalization combined with a trainable base embedding $E_{\mathrm{base}} \in \mathbb{R}^{K \times d}$:

$$
H_p^0 = H_c^0 = \operatorname{LN}(X_0 W_{\mathrm{init}} + b_{\mathrm{init}}) + E_{\mathrm{base}}
$$

At graph layer $l$, representations are updated using relational graph convolutions:

$$
\begin{aligned}
H_p^{l+1} &= \operatorname{GELU}\!\Big( \operatorname{LN}\!\big( H_p^l W_{p,0}^l + \bar{A}_{p,\rightarrow} H_p^l W_\rightarrow^l + \bar{A}_{p,\leftarrow} H_p^l W_\leftarrow^l \big) \Big) \\
H_c^{l+1} &= \operatorname{GELU}\!\Big( \operatorname{LN}\!\big( H_c^l W_{c,0}^l + \bar{A}_c H_c^l W_c^l \big) \Big)
\end{aligned}
$$

where:
* $\bar{A}_{p,\rightarrow} = \operatorname{row\_norm}(A_p)$ is the forward normalized prerequisite adjacency matrix,
* $\bar{A}_{p,\leftarrow} = \operatorname{row\_norm}(A_p^\top)$ is the reverse normalized prerequisite adjacency matrix,
* $\bar{A}_c = \operatorname{row\_norm}(A_c)$ is the normalized collaborative adjacency matrix,
* $W_{p,0}^l, W_\rightarrow^l, W_\leftarrow^l, W_{c,0}^l, W_c^l \in \mathbb{R}^{d \times d}$ are trainable projection matrices,
* $\operatorname{GELU}$ and $\operatorname{LN}$ denote the GELU activation function and Layer Normalization, respectively.

The forward and reverse projections in the prerequisite branch preserve prerequisite directionality.

---

### 1.1 View Fusion

To fuse representations from both views, let $h_j^p$ and $h_j^c$ denote the final prerequisite-view and collaborative-view representations of concept $c_j$. Their contributions are combined through learned view attention:

$$
\begin{aligned}
a_{j,v} &= w_a^\top h_j^v, \qquad v \in \{p, c\} \\
(\alpha_{j,p}, \alpha_{j,c}) &= \operatorname{softmax}(a_{j,p}, a_{j,c}) \\
e_j &= \alpha_{j,p} h_j^p + \alpha_{j,c} h_j^c
\end{aligned}
$$

where $w_a \in \mathbb{R}^d$ is a trainable attention projection vector, $\alpha_{j,v}$ is the normalized attention weight for view $v$, and $e_j \in \mathbb{R}^d$ is the fused representation of concept $c_j$.

---

### 1.2 Contrastive Alignment with False-Negative Masking

We align the two views using symmetric InfoNCE with false-negative masking. Let $\mathcal{N}_i^-$ contain concepts that share no edge with $c_i$ in either graph (forward prerequisite, reverse prerequisite, or collaborative), and let:

$$
s_{ij}^{pc} = \frac{\operatorname{sim}(h_i^p, h_j^c)}{\tau}
$$

where $\operatorname{sim}(\cdot,\cdot)$ denotes cosine similarity and $\tau > 0$ is the contrastive temperature parameter. The prerequisite-to-collaborative InfoNCE loss is computed as:

$$
\mathcal{L}_{p\rightarrow c} = -\frac{1}{K}\sum_{i=1}^{K} \log \frac{\exp(s_{ii}^{pc})} {\exp(s_{ii}^{pc}) + \sum_{j \in \mathcal{N}_i^-}\exp(s_{ij}^{pc})}
$$

The collaborative-to-prerequisite loss $\mathcal{L}_{c\rightarrow p}$ is defined symmetrically:

$$
\mathcal{L}_{c\rightarrow p} = -\frac{1}{K}\sum_{i=1}^{K} \log \frac{\exp(s_{ii}^{cp})} {\exp(s_{ii}^{cp}) + \sum_{j \in \mathcal{N}_i^-}\exp(s_{ij}^{cp})}
$$

and the total relation-learning objective is computed as the average:

$$
\mathcal{L}_{\mathrm{rel}} = \frac{1}{2}\left(\mathcal{L}_{p\rightarrow c} + \mathcal{L}_{c\rightarrow p}\right)
$$

---

### 1.3 Module Outputs

This module outputs the view-specific representations and the fused concept embeddings:
$$\{h_j^p\}_{j=1}^K, \quad \{h_j^c\}_{j=1}^K, \quad \{e_j\}_{j=1}^K$$

The high-confidence prerequisite graph $G_h$ is constructed upstream via empirical transition frequency thresholding and expert rules, and is utilized downstream by the candidate constructor and path decoder.

---

## 2. Code Verification and Discrepancy Analysis

| Component | Initial Description | Codebase Implementation | Code Reference |
| :--- | :--- | :--- | :--- |
| **View Attention** | $a_{j,v} = w_a^\top \tanh(W_v h_j^v + b_v)$ | Single linear layer without $\tanh$ or bias: $a_{j,v} = w_a^\top h_j^v$ | [`models/step1_knowledge_topology_encoder.py:67, 96-98`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L67-L98) |
| **Module Output** | Outputs fused embeddings $\{e_j\}$ and prerequisite graph $G_h$ | Returns `(explicit, implicit, fused)`. $G_h$ is constructed upstream in graph preprocessing, not generated by GCN. | [`models/step1_knowledge_topology_encoder.py:99`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L99) |
| **Reverse Adjacency** | Written as $\bar{A}_p^\top$ (transpose of row-normalized matrix) | Explicitly row-normalized reverse graph: $\operatorname{row\_norm}(A_p^\top)$ | [`models/step1_knowledge_topology_encoder.py:15-16, 76`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L15-L76) |
| **Relation Loss** | $\mathcal{L}_{\mathrm{rel}} = \mathcal{L}_{p\to c} + \mathcal{L}_{c\to p}$ | Averages the two directions: $0.5 \times (\mathcal{L}_{p\to c} + \mathcal{L}_{c\to p})$ | [`models/step1_knowledge_topology_encoder.py:111-112`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L111-L112) |
| **Initial Projection** | $H_p^0 = H_c^0 = X_0$ (assumes $d_0 = d$) | $X_0 \in \mathbb{R}^{K \times d_0}$ is projected via $\operatorname{Linear}(d_0, d) + \operatorname{LayerNorm}$ and added to learnable base embedding | [`models/step1_knowledge_topology_encoder.py:60-62, 83`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L60-L83) |
| **LayerNorm & GELU Order** | $\operatorname{GELU}(\operatorname{LN}(\dots))$ | Exactly matches `F.gelu(self.norm(value))` | [`models/step1_knowledge_topology_encoder.py:41-42`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L41-L42) |
| **False-Negative Mask** | Concepts with no edge in either graph | Exactly matches `(prerequisite + prerequisite.T + collaborative) > 0` | [`models/step1_knowledge_topology_encoder.py:19-24, 108-110`](file:///home/raed/TCLGM/models/step1_knowledge_topology_encoder.py#L19-L24) |

---

## 3. Raw LaTeX Snippet (For Direct Paper Inclusion)

```latex
For modeling concept representations, let $X_0 \in \mathbb{R}^{K \times d_0}$ contain the initial features of the $K$ concepts (e.g., character $n$-gram representations). The initial representations are obtained via a linear projection with layer normalization combined with a trainable base embedding $E_{\mathrm{base}} \in \mathbb{R}^{K \times d}$:
\begin{equation}
H_p^0 = H_c^0 = \operatorname{LN}(X_0 W_{\mathrm{init}} + b_{\mathrm{init}}) + E_{\mathrm{base}}.
\end{equation}
At graph layer $l$, representations are updated using relational graph convolutions:
\begin{equation}
\begin{aligned}
H_p^{l+1} = \operatorname{GELU}\!\Big( \operatorname{LN}\!\big( &H_p^l W_{p,0}^l + \bar{A}_{p,\rightarrow} H_p^l W_\rightarrow^l \\
&+ \bar{A}_{p,\leftarrow} H_p^l W_\leftarrow^l \big) \Big),
\end{aligned}
\label{eq:rgcnp}
\end{equation}
\begin{equation}
H_c^{l+1} = \operatorname{GELU}\!\left( \operatorname{LN}\!\left( H_c^l W_{c,0}^l + \bar{A}_c H_c^l W_c^l \right) \right),
\label{eq:rgcnc}
\end{equation}
where $\bar{A}_{p,\rightarrow} = \operatorname{row\_norm}(A_p)$, $\bar{A}_{p,\leftarrow} = \operatorname{row\_norm}(A_p^\top)$, and $\bar{A}_c = \operatorname{row\_norm}(A_c)$ are the row-normalized adjacency matrices; $W_{p,0}^l, W_\rightarrow^l, W_\leftarrow^l, W_{c,0}^l, W_c^l \in \mathbb{R}^{d \times d}$ are trainable projection matrices; and $\operatorname{GELU}$ and $\operatorname{LN}$ denote GELU activation and layer normalization. The forward and reverse projections in Eq.~\eqref{eq:rgcnp} preserve prerequisite directionality.

Then, to fuse representations from both views, let $h_j^p$ and $h_j^c$ denote the final prerequisite-view and collaborative-view representations of concept $c_j$. Their contributions are combined through learned view attention:
\begin{equation}
\begin{aligned}
a_{j,v} &= w_a^\top h_j^v, \qquad v \in \{p, c\}, \\
(\alpha_{j,p}, \alpha_{j,c}) &= \operatorname{softmax}(a_{j,p}, a_{j,c}), \\
e_j &= \alpha_{j,p} h_j^p + \alpha_{j,c} h_j^c,
\end{aligned}
\label{eq:viewattention}
\end{equation}
where $w_a \in \mathbb{R}^d$ is a trainable attention vector, $\alpha_{j,v}$ is the normalized attention weight for view $v$, and $e_j$ is the fused representation of $c_j$.

We align the two views using symmetric InfoNCE \cite{oord2018cpc} with false-negative masking. Let $\mathcal{N}_i^-$ contain concepts that share no edge with $c_i$ in either graph (forward prerequisite, reverse prerequisite, or collaborative), and let $s_{ij}^{pc} = \frac{\operatorname{sim}(h_i^p, h_j^c)}{\tau}$, where $\operatorname{sim}(\cdot,\cdot)$ denotes cosine similarity and $\tau > 0$ is the contrastive temperature. The prerequisite-to-collaborative loss is computed as:
\begin{equation}
\mathcal{L}_{p\rightarrow c} = -\frac{1}{K}\sum_{i=1}^{K} \log \frac{\exp(s_{ii}^{pc})} {\exp(s_{ii}^{pc}) + \sum_{j \in \mathcal{N}_i^-}\exp(s_{ij}^{pc})}.
\label{eq:infonce}
\end{equation}
The collaborative-to-prerequisite loss $\mathcal{L}_{c\rightarrow p}$ is defined symmetrically, and the total relation-learning objective is:
\begin{equation}
\mathcal{L}_{\mathrm{rel}} = \frac{1}{2}\left(\mathcal{L}_{p\rightarrow c} + \mathcal{L}_{c\rightarrow p}\right).
\label{eq:relationloss}
\end{equation}
This module outputs the view-specific and fused concept embeddings $\{h_j^p\}_{j=1}^K$, $\{h_j^c\}_{j=1}^K$, and $\{e_j\}_{j=1}^{K}$.
```
