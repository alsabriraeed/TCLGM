# TEMPO — Time-Constrained Learning Goal Management

Adaptive TEMPO: a time-budget-aware, prerequisite-enforcing curriculum recommendation system.

## Repository Structure

```
TCLGM/
├── main.py                          # Entry point: train | recommend | evaluate
├── config.py                        # Config loader with documented defaults
├── config.yaml                      # Default config (ASSISST2012)
├── requirements.txt
├── example_request.json             # Example inference request
│
├── configs/
│   └── config_ASSISST2017.yaml      # Config for ASSISST2017 dataset
│
├── models/                          # All 6-stage TEMPO model components
│   ├── step1_knowledge_topology_encoder.py
│   ├── step2_personalized_time_cost_model.py
│   ├── step3_budget_adaptive_candidate_constructor.py
│   ├── step4_budget_aware_path_decoder.py
│   ├── cognitive_simulator.py
│   ├── goal_manager.py
│   ├── learner_state_encoder.py
│   ├── kt_aux_head.py
│   ├── multi_critic.py
│   ├── greedy_anchor.py
│   └── online_planner.py
│
├── training/                        # Training routines for each stage
│   ├── constrained_optimization.py  # Stage 6: REINFORCE policy training
│   ├── train_topology.py            # Stage 3: GCN topology encoder
│   ├── train_time_model.py          # Stage 4: time cost model
│   ├── train_simulator.py           # Stage 5: cognitive simulator (DKT)
│   └── train_decoder.py
│
├── utils/
│   ├── graph_utils.py               # Graph construction (prerequisite + collaborative)
│   ├── data.py                      # Data loading, splitting, episode creation
│   ├── batching.py
│   ├── logging_utils.py
│   └── reproducibility.py
│
├── evaluation/
│   └── evaluate_tempo.py            # Core policy evaluation
│
└── artifacts/
    ├── ASSISST2012/                 # Pre-trained checkpoints + graphs (ASSISST2012)
    │   ├── checkpoints/             # 6 .pt model checkpoints
    │   ├── graphs/                  # prerequisite.npy, collaborative.npy, prerequisite_confidence.npy
    │   ├── concept_embeddings.npy
    │   ├── concept_difficulties.npy
    │   ├── concept_weights.npy
    │   ├── metadata.json
    │   ├── resolved_config.yaml
    │   └── interactions_processed.csv
    └── ASSISST2017/                 # Pre-trained checkpoints + graphs (ASSISST2017)
        └── ... (same structure)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Inference (Recommendation) — No training needed

Use a pre-trained artifact directory to generate a time-constrained learning path:

```bash
# ASSISST2012
python main.py recommend \
  --artifacts artifacts/ASSISST2012 \
  --request example_request.json \
  --device auto

# ASSISST2017
python main.py recommend \
  --artifacts artifacts/ASSISST2017 \
  --request example_request.json \
  --device auto
```

#### Request JSON format (`example_request.json`):
```json
{
  "concepts":      ["Concept A", "Concept B"],
  "correct":       [1, 0],
  "elapsed_sec":   [42.0, 85.0],
  "lag_sec":       [0.0, 3600.0],
  "targets":       ["Concept C"],
  "budget_seconds": 1800.0
}
```
- `concepts` / `correct` / `elapsed_sec` / `lag_sec`: learner history
- `targets`: learning goal concept names (must match concept names in `metadata.json`)
- `budget_seconds`: total available study time in seconds

---

### 2. Evaluation — Batch evaluation over test episodes

```bash
python main.py evaluate \
  --artifacts artifacts/ASSISST2012 \
  --budget-fraction 0.75 \
  --device auto
```

---

### 3. Training from scratch (6-stage pipeline)

Edit `config.yaml` to point `data_dir` at your dataset directory, then:

```bash
python main.py train \
  --config config.yaml \
  --output artifacts/MY_DATASET \
  --device auto
```

To resume from a specific stage (e.g. re-train only the policy decoder from stage 6):

```bash
python main.py train \
  --config config.yaml \
  --output artifacts/MY_DATASET \
  --from-stage 6 \
  --device auto
```

Training stages:
| Stage | Description |
|---|---|
| 1 | Load & temporally split interactions |
| 2 | Build prerequisite + collaborative graphs |
| 3 | Train GCN topology encoder |
| 4 | Train personalized time cost model |
| 5 | Train cognitive simulator (DKT) |
| 6 | Train REINFORCE policy decoder |

---

## Graph Construction

Graphs are built automatically during `train` (Stage 2) and saved to `<output>/graphs/`. 
Pre-built graphs are already included in `artifacts/ASSISST2012/graphs/` and `artifacts/ASSISST2017/graphs/`.

To rebuild graphs for an existing artifact without full retraining:
```bash
python main.py train \
  --config config.yaml \
  --output artifacts/ASSISST2012 \
  --from-stage 2 \
  --device auto
```
