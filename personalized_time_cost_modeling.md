# Step 2: Personalized Time-Cost Modeling

This document provides the mathematically verified description and code verification for the **Personalized Time-Cost Model** (Step 2 of TCLGM), directly aligned with the implementation in [`models/learner_state_encoder.py`](file:///home/raed/TCLGM/models/learner_state_encoder.py), [`models/step2_personalized_time_cost_model.py`](file:///home/raed/TCLGM/models/step2_personalized_time_cost_model.py), and [`training/train_time_model.py`](file:///home/raed/TCLGM/training/train_time_model.py).

---

## 1. Mathematical Formulation (Preview-Ready Markdown)

Learners differ in both prior knowledge and the time required to complete the same concept. TCLGM captures this heterogeneity by conditioning prospective concept-completion times on each learner's interaction history. 

For learner $u$, let $H_u = \{(q_n, a_n, \Delta_n, \ell_n)\}_{n=1}^{N_u}$ denote the interaction history, where:
* $N_u$ is the sequence length,
* $q_n \in C$ is the concept practiced at step $n$,
* $a_n \in \{0, 1\}$ is observed response correctness,
* $\Delta_n > 0$ is the active completion time (in seconds),
* $\ell_n \geq 0$ is the interaction lag since the preceding event.

### 1.1 Interaction Representation & History Encoder

The composite interaction token $\tilde{x}_n$ concatenates the fused concept embedding $e_{q_n} \in \mathbb{R}^d$, an 8-dimensional learned response embedding $E_a(a_n)$, and log-transformed continuous temporal features:

$$
\tilde{x}_n = \left[ e_{q_n} \;\Vert\; E_a(a_n) \;\Vert\; \log(1 + \Delta_n) \;\Vert\; \log(1 + \ell_n) \right] \in \mathbb{R}^{d + 8 + 2}
$$

where $\Vert$ denotes vector concatenation. This token is projected into the hidden space ($d_h = 64$) via a linear layer with Layer Normalization and GELU activation:

$$
x_n = \operatorname{GELU}\left(\operatorname{LN}\left(W_{\mathrm{in}}\tilde{x}_n + b_{\mathrm{in}}\right)\right) \in \mathbb{R}^{d_h}
$$

A Gated Recurrent Unit ($\operatorname{GRU}$) processes the sequence of interaction representations to capture temporal dynamics. The learner state $v_u \in \mathbb{R}^{d_h}$ is defined as the final hidden state for non-empty histories, or a trainable cold-start parameter vector $v_{\mathrm{cold}}$ when no prior interactions exist:

$$
v_u = \begin{cases}
\operatorname{GRU}(x_1, \ldots, x_{N_u}), & \text{if } N_u > 0, \\
v_{\mathrm{cold}}, & \text{if } N_u = 0.
\end{cases}
$$

---

### 1.2 Heteroscedastic Log-Normal Duration Distribution

Let $T_{u,j} > 0$ denote the random time required by learner $u$ to complete concept $c_j$. TCLGM models the conditional duration distribution as a heteroscedastic log-normal distribution:

$$
\log T_{u,j} \mid H_u, c_j \sim \mathcal{N}\left(\mu_{u,j}, \sigma_{u,j}^2\right)
$$

where $\mu_{u,j}$ and $\sigma_{u,j} > 0$ are the learner-concept location and scale parameters. 

To jointly capture non-linear learner–concept interactions, learner speed, and intrinsic concept difficulty, the model concatenates $[v_u \;\Vert\; e_j] \in \mathbb{R}^{d_h + d}$ and passes it through a shared 2-layer residual MLP trunk $g(v_u, e_j) \in \mathbb{R}^{d_{\mathrm{time}}}$. The parameters are computed as:

$$
\begin{aligned}
\mu_{u,j} &= \left( w_\mu^\top g(v_u, e_j) - f_{\mathrm{speed}}(v_u) + b_j^{\mathrm{concept}} \right) \cdot \sigma_{\log T} + \mu_{\log T} \\
\sigma_{u,j} &= \operatorname{clip}\!\left( \operatorname{softplus}\!\left( w_\sigma^\top g(v_u, e_j) \right) \cdot \sigma_{\log T}, \; 0.05, \; 3.0 \right)
\end{aligned}
$$

where:
* $w_\mu, w_\sigma \in \mathbb{R}^{d_{\mathrm{time}}}$ are linear projection heads for location and scale,
* $f_{\mathrm{speed}}(v_u) \in \mathbb{R}$ is a dedicated learner speed subnetwork (subtracted so that faster learners require less completion time),
* $b_j^{\mathrm{concept}} \in \mathbb{R}$ is a trainable concept-specific baseline duration bias,
* $\mu_{\log T}$ and $\sigma_{\log T}$ are the empirical mean and standard deviation of log-durations across the training set used for target standardization.

---

### 1.3 Prospective Cost Estimation

The prospective learner-specific cost $\hat{c}_{u,j}$ used for candidate construction (Step 3) and budget-aware path decoding (Step 4) is the expected completion time under the log-normal model:

$$
\hat{c}_{u,j} = \mathbb{E}\left[T_{u,j} \mid H_u, c_j\right] = \exp\!\left( \mu_{u,j} + \frac{1}{2}\sigma_{u,j}^2 \right)
$$

---

### 1.4 Training Objective

Given a training set $\mathcal{D}_{\mathrm{time}} = \{(u_i, c_{j_i}, y_i)\}_{i=1}^{N_{\mathrm{time}}}$ of observed active completion times $y_i > 0$, the learner state encoder and time-cost model are trained jointly by minimizing the Gaussian negative log-likelihood:

$$
\mathcal{L}_{\mathrm{time}} = \frac{1}{N_{\mathrm{time}}} \sum_{i=1}^{N_{\mathrm{time}}} \left[ \frac{(\log y_i - \mu_i)^2}{2\sigma_i^2} + \log \sigma_i \right]
$$

---

## 2. Encoder Architecture & Component Inventory

| Component / Submodule | Class / PyTorch Implementation | Input $\rightarrow$ Output Shape | Description |
| :--- | :--- | :--- | :--- |
| **Response Encoder ($E_a$)** | `nn.Embedding(2, 8)` in `LearnerStateEncoder` | $\{0, 1\} \rightarrow \mathbb{R}^8$ | Learned embedding lookup for binary correctness. |
| **Temporal Features** | $\log(1 + \Delta_n)$, $\log(1 + \ell_n)$ | $\mathbb{R}^+ \rightarrow \mathbb{R}^2$ | Deterministic logarithmic transform of elapsed time and lag. |
| **Interaction Projector** | `self.input_projection = nn.Sequential(...)` | $\mathbb{R}^{d + 8 + 2} \rightarrow \mathbb{R}^{64}$ | Linear(74, 64) $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Dropout(0.1). |
| **History Encoder ($\operatorname{Enc}_H$)** | `self.gru = nn.GRU(64, 64, batch_first=True)` | $\mathbb{R}^{N_u \times 64} \rightarrow \mathbb{R}^{64}$ | Gated Recurrent Unit extracting last hidden state $h_{N_u}$. |
| **Cold-Start Representation** | `self.cold_start = nn.Parameter(...)` | $\mathbb{R}^{64}$ | Trainable baseline state vector for empty learner history. |
| **Residual Backbone** | `self.trunk = nn.Sequential(ResidualBlock, ResidualBlock)` | $\mathbb{R}^{128} \rightarrow \mathbb{R}^{128}$ | Two pre-LayerNorm residual blocks with GELU activations. |
| **Learner Speed Net ($f_{\mathrm{speed}}$)**| `self.speed = nn.Sequential(...)` | $\mathbb{R}^{64} \rightarrow \mathbb{R}^1$ | Linear(64, 32) $\rightarrow$ GELU $\rightarrow$ Linear(32, 1). |
| **Concept Duration Bias ($b_j$)** | `self.concept_bias = nn.Embedding(K, 1)` | Concept ID $\rightarrow \mathbb{R}^1$ | Inherent concept-level time intensity offset. |
| **Mean & Scale Heads** | `self.mean_head`, `self.scale_head` | $\mathbb{R}^{128} \rightarrow \mathbb{R}^1$ | Linear heads producing standardized parameters. |

---

## 3. Discrepancy & Code Verification Analysis

1. **Temporal Features ($E_\Delta, E_\ell$ vs. Log-Transforms):**
   * *Draft:* Stated that $E_\Delta$ and $E_\ell$ are learned encoders.
   * *Code:* [`models/learner_state_encoder.py:28`](file:///home/raed/TCLGM/models/learner_state_encoder.py#L28) uses `torch.stack((torch.log1p(elapsed), torch.log1p(lag)), dim=-1)`. They are deterministic log-transforms, not neural encoders.
2. **History Encoder Architecture:**
   * *Draft:* Left abstract as $\operatorname{Enc}_H$.
   * *Code:* [`models/learner_state_encoder.py:20`](file:///home/raed/TCLGM/models/learner_state_encoder.py#L20) explicitly uses `nn.GRU`.
3. **Structured Decomposition of $\mu_{u,j}$:**
   * *Draft:* Formulated as generic $f_\mu(v_u, e_j)$.
   * *Code:* [`models/step2_personalized_time_cost_model.py:46–50`](file:///home/raed/TCLGM/models/step2_personalized_time_cost_model.py#L46-L50) explicitly subtracts learner speed ($-\mathrm{speed}(v_u)$) and adds concept bias ($+b_j^{\mathrm{concept}}$), then rescales by $\sigma_{\log T}$ and shifts by $\mu_{\log T}$.
4. **Scale Parameter Bounding:**
   * *Draft:* $\operatorname{softplus}(\dots) + \epsilon_\sigma$.
   * *Code:* [`models/step2_personalized_time_cost_model.py:51–52`](file:///home/raed/TCLGM/models/step2_personalized_time_cost_model.py#L51-L52) applies `softplus`, multiplies by $\sigma_{\log T}$, and clamps to $[0.05, 3.0]$.
5. **Expected Time $\hat{c}_{u,j}$ & Loss $\mathcal{L}_{\mathrm{time}}$:**
   * Both match the implementation in [`models/step2_personalized_time_cost_model.py:60, 67`](file:///home/raed/TCLGM/models/step2_personalized_time_cost_model.py#L60-L67) exactly.

---

## 4. Raw LaTeX Snippet (For Direct Paper Inclusion)

```latex
\subsubsection{Personalized Time-Cost Modeling}
\label{subsec:time}
Learners differ in both prior knowledge and the time required to complete the same concept. TCLGM captures this heterogeneity by conditioning prospective concept-completion times on each learner's interaction history. For learner $u$, let $H_u = \{(q_n, a_n, \Delta_n, \ell_n)\}_{n=1}^{N_u}$ denote the interaction history, where $N_u$ is sequence length, $q_n \in C$ is the concept practiced at step $n$, $a_n \in \{0, 1\}$ is observed correctness, $\Delta_n > 0$ is active completion time, and $\ell_n \geq 0$ is the time lag since the preceding interaction.

The interaction representation $\tilde{x}_n$ combines the fused concept embedding $e_{q_n} \in \mathbb{R}^d$, a learned correctness embedding $E_a(a_n) \in \mathbb{R}^8$, and log-transformed temporal features:
\begin{equation}
\tilde{x}_n = e_{q_n} \mathbin{\Vert} E_a(a_n) \mathbin{\Vert} \log(1 + \Delta_n) \mathbin{\Vert} \log(1 + \ell_n),
\label{eq:learnerrawinput}
\end{equation}
where $\mathbin{\Vert}$ denotes vector concatenation. This composite token is mapped into the hidden space via a projection layer:
\begin{equation}
x_n = \operatorname{GELU}(\operatorname{LN}(W_{\mathrm{in}}\tilde{x}_n + b_{\mathrm{in}})).
\label{eq:learnerinput}
\end{equation}
A Gated Recurrent Unit (GRU) summarizes the interaction sequence into a learner-specific state $v_u \in \mathbb{R}^{d_h}$:
\begin{equation}
v_u = \begin{cases}
\operatorname{GRU}(x_1, \ldots, x_{N_u}), & N_u > 0, \\
v_{\mathrm{cold}}, & N_u = 0,
\end{cases}
\label{eq:learnerstate}
\end{equation}
where $v_{\mathrm{cold}} \in \mathbb{R}^{d_h}$ is a trainable parameter vector supporting cold-start prediction for learners without prior interactions.

Let $T_{u,j} > 0$ denote the random time required by learner $u$ to complete concept $c_j$. TCLGM models the conditional duration distribution as a heteroscedastic log-normal distribution:
\begin{equation}
\log T_{u,j} \mid H_u, c_j \sim \mathcal{N}\left(\mu_{u,j}, \sigma_{u,j}^2\right),
\label{eq:lognormal}
\end{equation}
where $\mu_{u,j}$ and $\sigma_{u,j} > 0$ are the learner-concept location and scale parameters. To capture non-linear learner--concept interactions, learner speed, and intrinsic concept duration, the model passes $[v_u \mathbin{\Vert} e_j]$ through a shared residual MLP backbone $g(v_u, e_j) \in \mathbb{R}^{d_{\mathrm{time}}}$ and computes the parameters as:
\begin{equation}
\begin{aligned}
\mu_{u,j} &= \left( w_\mu^\top g(v_u, e_j) - f_{\mathrm{speed}}(v_u) + b_j^{\mathrm{concept}} \right) \cdot \sigma_{\log T} + \mu_{\log T}, \\
\sigma_{u,j} &= \operatorname{clip}\!\Big( \operatorname{softplus}\!\left( w_\sigma^\top g(v_u, e_j) \right) \cdot \sigma_{\log T}, \, 0.05, \, 3.0 \Big),
\end{aligned}
\label{eq:timeparameters}
\end{equation}
where $f_{\mathrm{speed}}$ is a dedicated learner speed subnetwork, $b_j^{\mathrm{concept}}$ is a trainable concept-specific baseline duration bias, and $\mu_{\log T}, \sigma_{\log T}$ denote the empirical mean and standard deviation of log-durations across the training set.

The expected completion time is used as the prospective learner-specific cost $\hat{c}_{u,j}$ for candidate construction and path decoding:
\begin{equation}
\hat{c}_{u,j} = \mathbb{E}[T_{u,j} \mid H_u, c_j] = \exp\!\left( \mu_{u,j} + \frac{1}{2}\sigma_{u,j}^2 \right).
\label{eq:expectedcost}
\end{equation}

Given a training set $\mathcal{D}_{\mathrm{time}} = \{(u_i, c_{j_i}, y_i)\}_{i=1}^{N_{\mathrm{time}}}$ of observed active durations $y_i > 0$, the learner state encoder and time-cost model are trained jointly by minimizing the Gaussian negative log-likelihood:
\begin{equation}
\mathcal{L}_{\mathrm{time}} = \frac{1}{N_{\mathrm{time}}} \sum_{i=1}^{N_{\mathrm{time}}} \left[ \frac{(\log y_i - \mu_i)^2}{2\sigma_i^2} + \log \sigma_i \right].
\label{eq:timeloss}
\end{equation}
```
