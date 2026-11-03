# Step 3: Adaptive Goals and Budget-Adaptive Candidate Construction

This document provides the mathematically verified description, component inventory, and code verification for **Adaptive Goal Formulation and Candidate Construction** (Step 3 of TCLGM), directly aligned with [`models/goal_manager.py`](file:///home/raed/TCLGM/models/goal_manager.py), [`models/step3_budget_adaptive_candidate_constructor.py`](file:///home/raed/TCLGM/models/step3_budget_adaptive_candidate_constructor.py), and [`models/step4_budget_aware_path_decoder.py`](file:///home/raed/TCLGM/models/step4_budget_aware_path_decoder.py).

---

## 1. Mathematical Formulation (Preview-Ready Markdown)

TCLGM combines an interpretable pedagogical goal-priority function with graph-closure operators and budget-aware candidate filtering. This ensures that recommended learning paths balance target concept mastery, prerequisite readiness, and personalized review needs within the learner's available study time.

### 1.1 Pedagogical Goal Priority Formulation

For learner $u$, let $T_u \subseteq C$ denote the explicit target concept set, and let $m_u^{(0)} \in [0, 1]^K$ denote the initial mastery vector. The mastery gap for concept $c_j$ is defined as:

$$
\operatorname{gap}_{u,j} = \operatorname{clip}\left(1 - m_{u,j}^{(0)}, \; 0, \; 1\right)
$$

The target relevance $r_{u,j}$ measures proximity to target concepts along the prerequisite graph $G_p$, discounted by graph distance:

$$
r_{u,j} = \max_{c_k \in T_u} \gamma_{\mathrm{anc}}^{\operatorname{dist}_{G_p}(c_j, c_k)}
$$

where $\gamma_{\mathrm{anc}} \in (0, 1]$ (default $0.80$) is the ancestor discount factor.

To adaptively balance initial learning against review needs, TCLGM computes a dynamic review probability $p_{\mathrm{rev}} \in [0, 1]$ from historical interaction signals:

$$
p_{\mathrm{rev}} = \operatorname{clip}\left(0.45 \cdot \text{cov}(T_u) + 0.35 \cdot \text{acc}_{\mathrm{recent}} + 0.20 \cdot \text{press}_{\mathrm{forget}}, \; 0, \; 1\right)
$$

where $\text{cov}(T_u)$ is target concept historical coverage, $\text{acc}_{\mathrm{recent}}$ is recent response accuracy, and $\text{press}_{\mathrm{forget}}$ is the mean forgetting pressure across previously studied concepts.

The priority $G_{u,j}$ of concept $c_j$ is computed by blending target acquisition, review urgency, and curriculum progression:

$$
G_{u,j} = w_j \cdot \Big[ (1 - p_{\mathrm{rev}}) \cdot r_{u,j} \cdot \operatorname{gap}_{u,j} + p_{\mathrm{rev}} \cdot \mathbb{I}[s_{u,j} > 0] \cdot \operatorname{gap}_{u,j} \cdot \max(f_{u,j}, 0.05) + w_{\mathrm{curr}} \cdot \operatorname{gap}_{u,j} \Big] + \beta_{\mathrm{target}} \cdot \mathbb{I}[c_j \in T_u] \cdot w_j \cdot \operatorname{gap}_{u,j}
$$

where:
* $w_j > 0$ is the intrinsic concept importance weight,
* $s_{u,j}$ is the number of times learner $u$ has previously studied concept $c_j$,
* $f_{u,j} \in [0, 1]$ is the estimated forgetting probability,
* $w_{\mathrm{curr}}$ (default $0.15$) weights broader curriculum progression,
* $\beta_{\mathrm{target}} = 3.0$ is a high-gain prioritization bonus assigned directly to target concepts.

Concepts with $s_{u,j} > 0$ and $m_{u,j}^{(0)} < \theta_m$ are ranked by $G_{u,j}$ to identify a top-$k$ review goal set $R_u$ ($|R_u| \leq K_{\mathrm{rev}}$).

---

### 1.2 Unmet Prerequisite Closure & Affordability

Let $M_u^{(0)} = \{c_j : m_{u,j}^{(0)} \geq \theta_m\}$ be the initially mastered concept set, where $\theta_m \in [0, 1]$ (default $0.85$) is the mastery threshold. 

For any concept $c_j$, its unmet prerequisite closure $\mathcal{Q}_u(j)$ in the high-confidence prerequisite DAG $G_h$ contains $c_j$ and all its unmastered transitive ancestors:

$$
\mathcal{Q}_u(j) = \left( \{c_j\} \cup \operatorname{Anc}_{G_h}(c_j) \right) \setminus M_u^{(0)}
$$

A concept $c_j$ is closure-affordable if the predicted completion time of its entire unmet closure does not exceed the total budget $B_u$:

$$
\sum_{c_k \in \mathcal{Q}_u(j)} \hat{c}_{u,k} \leq B_u
$$

---

### 1.3 Priority-Density-Ranked Candidate Construction

Rather than evaluating the entire concept universe $C$, TCLGM constructs a focused candidate pool $C_u$ ($|C_u| \leq N_{\max}$, default $30$–$40$) via cost-normalized priority density ranking:

1. **Mandatory Candidate Retention:** The unmet closure of all target concepts is retained first:
   $$
   C_{\mathrm{mand}} = \bigcup_{c_j \in T_u} \mathcal{Q}_u(j)
   $$
2. **Priority Density Ranking:** Mandatory concepts and optional review seeds $c_j \in R_u$ are ranked by their priority density:
   $$
   \rho(c_j) = \frac{G_{u,j}}{\sqrt{\hat{c}_{u,j}}}
   $$
   Filling available pool capacity with top-ranking items.
3. **Curriculum Expansion:** If remaining capacity exists, unmastered concepts from the broader curriculum with affordable closures are admitted in decreasing order of linear density $G_{u,j} / \hat{c}_{u,j}$.
4. **Prerequisite Completeness Pruning:** To guarantee that no candidate is exposed with missing prerequisites, TCLGM iteratively prunes any concept whose unmet predecessors are not also present in $C_u$:
   $$
   \forall c_j \in C_u, \quad \operatorname{Pred}_{G_h}(c_j) \setminus M_u^{(0)} \subseteq C_u
   $$

The candidate pool $C_u$ serves as the discrete action space for the path decoder.

---

### 1.4 Step-Wise Remaining Budget and Feasible Action Frontier

At path step $t$, let $P^{<t} = (p_1, \ldots, p_{t-1})$ denote the prefix of selected concepts. The remaining study budget is dynamically updated as:

$$
b_t = B_u - \sum_{r=1}^{t-1} \hat{c}_{u, p_r}
$$

A candidate $c_j \in C_u \setminus P^{<t}$ is admissible at step $t$ if and only if all its immediate high-confidence predecessors are already mastered or completed, and its prospective cost fits within the remaining time:

$$
F_t = \left\{ c_j \in C_u \setminus P^{<t} : \operatorname{Pred}_{G_h}(c_j) \subseteq M_u^{(0)} \cup P^{<t}, \quad \hat{c}_{u,j} \leq b_t \right\}
$$

During path generation, the policy pointer distribution is strictly masked over $F_t$, guaranteeing that every selected concept is pedagogically ready and temporally feasible.

---

## 2. Component Inventory & Code References

| Component | Class / Method | Implementation Details | File Reference |
| :--- | :--- | :--- | :--- |
| **Adaptive Goal Manager** | `AdaptiveGoalManager.formulate()` | Calculates review probability $p_{\mathrm{rev}}$, target relevance $r_{u,j}$, and composite priority $G_{u,j}$ | [`models/goal_manager.py:43–91`](file:///home/raed/TCLGM/models/goal_manager.py#L43-L91) |
| **Prerequisite Closure** | `BudgetAdaptiveCandidateConstructor._closure()` | Traverses $G_h$ ancestors recursively, filtering out concepts already in $M_u^{(0)}$ | [`models/step3_budget_adaptive_candidate_constructor.py:43–50`](file:///home/raed/TCLGM/models/step3_budget_adaptive_candidate_constructor.py#L43-L50) |
| **Priority Density Ranking** | `BudgetAdaptiveCandidateConstructor.construct()` | Ranks candidates using $G_{u,j} / \sqrt{\hat{c}_{u,j}}$ and $G_{u,j} / \hat{c}_{u,j}$ | [`models/step3_budget_adaptive_candidate_constructor.py:68–100`](file:///home/raed/TCLGM/models/step3_budget_adaptive_candidate_constructor.py#L68-L100) |
| **Prerequisite Completeness Pruning** | `BudgetAdaptiveCandidateConstructor.construct()` | Iterative `while changed:` loop pruning candidates with missing unmet ancestors | [`models/step3_budget_adaptive_candidate_constructor.py:105–119`](file:///home/raed/TCLGM/models/step3_budget_adaptive_candidate_constructor.py#L105-L119) |
| **Local Action Graph** | `local_prerequisites` | Subgraph slice of $G_h$ on candidates, zeroing mastered predecessor rows | [`models/step3_budget_adaptive_candidate_constructor.py:129–135`](file:///home/raed/TCLGM/models/step3_budget_adaptive_candidate_constructor.py#L129-L135) |
| **Dynamic Frontier Mask** | `_prerequisite_ready` & `forward()` | Evaluates `(~selected) & ready & (cand_cost <= remaining)` | [`models/step4_budget_aware_path_decoder.py:42–45, 82–88`](file:///home/raed/TCLGM/models/step4_budget_aware_path_decoder.py#L42-L88) |

---

## 3. Discrepancy & Code Verification Analysis

1. **Goal Priority Formulation ($G_{u,j}$) — Major Mismatch:**
   * *Draft:* Stated that $G_{u,j} = f_G(v_u, e_j, x_{T_u}, m_{u,j}^{(0)}, h_{u,j}^{\mathrm{hist}}, \mathbb{I}[c_j \in T_u])$ where $f_G$ is a "trainable scoring network".
   * *Code:* [`models/goal_manager.py:68–77`](file:///home/raed/TCLGM/models/goal_manager.py#L68-L77) implements an explicit, interpretable pedagogical scoring algorithm (inspired by HA-LPR). It combines ancestor distance decay along $G_p$, mastery gap, review probability $p_{\mathrm{rev}}$, curriculum progression, and target bonuses. It is **not** a neural network.
   * *Where Target Context is Actually Used:* The attention-pooled target embedding $x_{T_u}$ is used in **Step 4** (the pointer decoder network: `target_score(target_context)` in [`models/step4_budget_aware_path_decoder.py:109`](file:///home/raed/TCLGM/models/step4_budget_aware_path_decoder.py#L109)), not inside a neural goal manager.

2. **Total Candidate Pool Budget Constraint ($\sum_{c_k \in C_u} \hat{c}_{u,k} \leq B_u$) — Factual Error:**
   * *Draft:* Stated that the sum of costs across the entire candidate pool $C_u$ is bounded by $B_u$.
   * *Code:* The candidate pool $C_u$ contains up to $N_{\max} = 30$ or $40$ candidate concepts. The sum of costs of *all* items in the pool is **not** restricted to $\leq B_u$ (otherwise the policy decoder would have no diverse candidate choices to optimize within budget). Instead, **individual admitted closures** are affordable, and the **selected trajectory** $P = (p_1, \ldots, p_L)$ is constrained by $B_u$ via step-wise budget tracking.

3. **Priority Density Ranking Metric:**
   * *Code:* Candidates are admitted by ranking concepts according to **priority density** ($\rho(c_j) = G_{u,j} / \sqrt{\hat{c}_{u,j}}$ for mandatory/review, and $G_{u,j} / \hat{c}_{u,j}$ for expansion). This cost-normalization ensures that expensive concepts must deliver proportionally higher pedagogical priority.

4. **Iterative Prerequisite Completeness Pruning:**
   * *Code:* Lines 105–119 of `BudgetAdaptiveCandidateConstructor` perform iterative transitive closure pruning to ensure that if any prerequisite of a concept was omitted, the concept itself is pruned. This prevents invalid action exposure.

5. **Feasible Action Frontier ($F_t$) and Remaining Budget ($b_t$) — 100% Match:**
   * The definitions of $b_t = B_u - \sum_{r=1}^{t-1} \hat{c}_{u, p_r}$ and $F_t$ matching `ready & (~selected) & (cost <= remaining)` are an **exact match** with lines 42–45 and 82–88 of [`models/step4_budget_aware_path_decoder.py`](file:///home/raed/TCLGM/models/step4_budget_aware_path_decoder.py).

---

## 4. Raw LaTeX Snippet (For Direct Paper Inclusion)

```latex
\subsection{Adaptive Goals and Candidate Construction}
\label{subsec:candidates}
TCLGM combines an interpretable pedagogical goal-priority function with graph-closure operators and budget-adaptive candidate filtering. For learner $u$, let $T_u \subseteq C$ denote the explicit target concept set, and let $m_u^{(0)} \in [0, 1]^K$ denote the initial mastery vector. The mastery gap for concept $c_j$ is defined as $\operatorname{gap}_{u,j} = \operatorname{clip}(1 - m_{u,j}^{(0)}, 0, 1)$, and its prerequisite target relevance is measured by ancestor distance discounting along $G_p$:
\begin{equation}
r_{u,j} = \max_{c_k \in T_u} \gamma_{\mathrm{anc}}^{\operatorname{dist}_{G_p}(c_j, c_k)},
\label{eq:targetrelevance}
\end{equation}
where $\gamma_{\mathrm{anc}} \in (0, 1]$ is the ancestor discount factor. To balance new learning against review needs, TCLGM dynamically computes a review probability $p_{\mathrm{rev}} \in [0, 1]$ from target coverage, recent accuracy, and forgetting pressure. The priority $G_{u,j}$ of concept $c_j$ is then computed as:
\begin{equation}
\begin{aligned}
G_{u,j} = w_j \cdot \Big[ &(1 - p_{\mathrm{rev}}) \cdot r_{u,j} \cdot \operatorname{gap}_{u,j} \\
&+ p_{\mathrm{rev}} \cdot \mathbb{I}[s_{u,j} > 0] \cdot \operatorname{gap}_{u,j} \cdot \max(f_{u,j}, 0.05) \\
&+ w_{\mathrm{curr}} \cdot \operatorname{gap}_{u,j} \Big] + \beta_{\mathrm{target}} \cdot \mathbb{I}[c_j \in T_u] \cdot w_j \cdot \operatorname{gap}_{u,j},
\end{aligned}
\label{eq:priority}
\end{equation}
where $w_j$ is the concept importance weight, $s_{u,j}$ is past study frequency, $f_{u,j}$ is estimated forgetting probability, $w_{\mathrm{curr}}$ weights curriculum progression, and $\beta_{\mathrm{target}}$ is a direct target prioritization bonus.

Let $M_u^{(0)} = \{c_j : m_{u,j}^{(0)} \geq \theta_m\}$ be the initially mastered concept set, where $\theta_m \in [0, 1]$ is the mastery threshold. The unmet prerequisite closure of concept $c_j$ in $G_h$ is defined as:
\begin{equation}
\mathcal{Q}_u(j) = \left( \{c_j\} \cup \operatorname{Anc}_{G_h}(c_j) \right) \setminus M_u^{(0)},
\label{eq:conceptclosure}
\end{equation}
where $\operatorname{Anc}_{G_h}(c_j)$ denotes all transitive prerequisite ancestors of $c_j$ in $G_h$. A concept is closure-affordable if the predicted completion time of its entire unmet closure does not exceed budget $B_u$: $\sum_{c_k \in \mathcal{Q}_u(j)} \hat c_{u,k} \leq B_u$.

TCLGM constructs a candidate pool $C_u$ ($|C_u| \leq N_{\max}$) by admitting the unmet closures of target concepts, top review goals, and curriculum expansion concepts ranked by cost-normalized priority density $\rho(c_j) = G_{u,j} / \sqrt{\hat c_{u,j}}$. An iterative pruning procedure removes any concept whose unmet predecessors were not retained, guaranteeing that $C_u$ is self-contained under unmet prerequisites.

At path step $t$, let $P^{<t} = (p_1, \ldots, p_{t-1})$ denote the selected prefix. The remaining predicted study time is:
\begin{equation}
b_t = B_u - \sum_{r=1}^{t-1} \hat c_{u, p_r}.
\label{eq:remainingtime}
\end{equation}
Let $\operatorname{Pred}_h(c_j)$ denote the immediate predecessors of $c_j$ in $G_h$. The feasible action frontier is:
\begin{equation}
F_t = \left\{ c_j \in C_u \setminus P^{<t} : \operatorname{Pred}_h(c_j) \subseteq M_u^{(0)} \cup P^{<t}, \quad \hat c_{u,j} \leq b_t \right\}.
\label{eq:frontier}
\end{equation}
Therefore, every concept exposed to the path decoder has its high-confidence prerequisites satisfied and can be completed within the remaining predicted study time.
```
