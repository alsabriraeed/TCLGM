"""Adaptive TEMPO end-to-end training and recommendation entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from config import load_config
from evaluation.evaluate_tempo import evaluate_policy
from models.cognitive_simulator import TimeEmittingDKT
from models.learner_state_encoder import LearnerStateEncoder
from models.online_planner import AdaptiveTEMPOPlanner
from models.step2_personalized_time_cost_model import PersonalizedTimeCostModel
from models.step4_budget_aware_path_decoder import BudgetAwarePathDecoder
from training.constrained_optimization import train_constrained_policy
from training.train_simulator import train_simulator
from training.train_time_model import train_time_cost_model
from training.train_topology import train_topology_encoder
from utils.data import (
    concept_difficulties, concept_weights, load_interactions, policy_episodes,
    subsample_by_learners, time_samples,
)
from utils.graph_utils import build_graphs, hashed_name_features, save_graphs
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed


log = get_logger("adaptive_tempo.main")


def _save_state(model: torch.nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def _resolve_artifact(filename: str, output_dir: Path, upstream_dir: Path) -> Path:
    target = output_dir / filename
    if target.exists():
        return target
    upstream = upstream_dir / filename
    if upstream.exists():
        return upstream
    return target


def train_pipeline(args) -> None:
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    requested_device = getattr(args, "device", None)
    if requested_device and requested_device != "auto":
        device = requested_device
        cfg["device"] = device
    else:
        device = cfg["device"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    from_stage = getattr(args, "from_stage", 1)
    to_stage = getattr(args, "to_stage", 6)

    upstream_dir_arg = getattr(args, "upstream", None)
    if upstream_dir_arg:
        upstream_dir = Path(upstream_dir_arg).resolve()
    elif cfg.get("upstream_artifacts"):
        upstream_dir = Path(cfg["upstream_artifacts"]).resolve()
    else:
        upstream_dir = output.resolve()
    ckpt_upstream = upstream_dir / "checkpoints"

    if getattr(args, "epochs", None) is not None:
        cfg.setdefault("decoder", {})["epochs"] = int(args.epochs)
    if getattr(args, "topology_epochs", None) is not None:
        cfg.setdefault("graph", {})["topology_epochs"] = int(args.topology_epochs)
    if getattr(args, "time_epochs", None) is not None:
        cfg.setdefault("time_model", {})["epochs"] = int(args.time_epochs)
    if getattr(args, "time_patience", None) is not None:
        cfg.setdefault("time_model", {})["early_stopping_patience"] = int(args.time_patience)
    if getattr(args, "budget_fractions", None) is not None:
        cfg.setdefault("training", {})["budget_fractions"] = [float(x) for x in args.budget_fractions]

    # Resolve the interactions CSV: explicit --data flag takes priority,
    # otherwise fall back to <data_dir>/processed/interactions_processed.csv
    # from the config file or upstream directory.
    if args.data:
        data_path = Path(args.data)
    else:
        data_dir = cfg.get("data_dir")
        if data_dir and (Path(data_dir) / "processed" / "interactions_processed.csv").exists():
            data_path = Path(data_dir) / "processed" / "interactions_processed.csv"
        else:
            data_path = _resolve_artifact("interactions_processed.csv", output, upstream_dir)
    log.info("Using interaction data from: %s", data_path)

    existing_metrics = {}
    metrics_path = _resolve_artifact("metrics.json", output, upstream_dir)
    if metrics_path.exists():
        try:
            existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing_metrics = {}

    if from_stage <= 1:
        log.info("Stage 1/6: loading and temporally splitting interactions")
        df, metadata = load_interactions(data_path, cfg)
        df.to_csv(output / "interactions_processed.csv", index=False)
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        n_concepts = int(metadata["n_concepts"])
    else:
        log.info("Skipping Stage 1 (loading existing processed interactions and metadata from %s)", upstream_dir)
        df = pd.read_csv(_resolve_artifact("interactions_processed.csv", output, upstream_dir))
        metadata = json.loads(_resolve_artifact("metadata.json", output, upstream_dir).read_text(encoding="utf-8"))
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        n_concepts = int(metadata["n_concepts"])

    # Quick Train Mode: Subsample interactions by learners if specified
    sample_fraction = getattr(args, "sample_fraction", None)
    max_learners = getattr(args, "max_learners", None)
    if (sample_fraction is not None and 0.0 < float(sample_fraction) < 1.0) or (max_learners is not None and int(max_learners) > 0):
        u_col = "uidx" if "uidx" in df.columns else "user_id"
        n_before = df[u_col].nunique()
        log.info("Quick Train Mode: subsampling interactions at learner level (sample_fraction=%s, max_learners=%s)",
                 sample_fraction, max_learners)
        df = subsample_by_learners(df, sample_fraction=sample_fraction, max_learners=max_learners, seed=int(cfg.get("seed", 42)))
        log.info("Subsampled dataset: %d rows across %d learners (reduced from %d learners)", len(df), df[u_col].nunique(), n_before)

    if from_stage <= 2:
        log.info("Stage 2/6: building confidence-scored prerequisite and collaborative graphs")
        prerequisite, prerequisite_confidence, collaborative = build_graphs(
            df, n_concepts, cfg, prerequisite_csv=args.prerequisites,
            concept_to_idx=metadata["concept_to_idx"])
        save_graphs(output / "graphs", prerequisite, prerequisite_confidence, collaborative)
    else:
        log.info("Skipping Stage 2 (loading existing graphs from %s)", upstream_dir)
        graphs_dir = (output / "graphs") if (output / "graphs").exists() else (upstream_dir / "graphs")
        prerequisite = np.load(graphs_dir / "prerequisite.npy")
        prerequisite_confidence = np.load(graphs_dir / "prerequisite_confidence.npy")
        collaborative = np.load(graphs_dir / "collaborative.npy")

    if from_stage <= 3:
        log.info("Stage 3/6: training directed dual-view topology encoder")
        initial_features = torch.tensor(
            hashed_name_features(metadata["idx_to_concept"]), dtype=torch.float32)
        topology, (_, _, embeddings) = train_topology_encoder(
            prerequisite, collaborative, cfg, device, initial_features)
        embeddings = embeddings.detach().to(device)
        _save_state(topology, checkpoint_dir / "topology_encoder.pt")
        np.save(output / "concept_embeddings.npy", embeddings.cpu().numpy())
    else:
        log.info("Skipping Stage 3 (loading existing concept embeddings from %s)", upstream_dir)
        emb_path = _resolve_artifact("concept_embeddings.npy", output, upstream_dir)
        embeddings = torch.tensor(
            np.load(emb_path),
            dtype=torch.float32, device=device)

    max_history = int(cfg["data"]["max_history"])
    if from_stage <= 4:
        log.info("Stage 4/6: training and conformally calibrating personalized time model")
        max_train_samples = int(cfg.get("time_model", {}).get("max_train_samples", 100000))
        max_val_samples = int(cfg.get("time_model", {}).get("max_val_samples", 20000))
        train_time = time_samples(df, "train", max_history, max_samples=max_train_samples, seed=int(cfg.get("seed", 42)))
        val_time = time_samples(df, "val", max_history, max_samples=max_val_samples, seed=int(cfg.get("seed", 42)))
        if not val_time:
            val_time = time_samples(df, "test", max_history, max_samples=max_val_samples, seed=int(cfg.get("seed", 42)))
        learner_encoder, time_model, time_metrics = train_time_cost_model(
            train_time, val_time, embeddings, cfg, device)
        _save_state(learner_encoder, checkpoint_dir / "learner_encoder.pt")
        _save_state(time_model, checkpoint_dir / "time_model.pt")
    else:
        log.info("Skipping Stage 4 (loading existing time model and learner encoder from %s)", upstream_dir)
        learner_ckpt = (checkpoint_dir / "learner_encoder.pt") if (checkpoint_dir / "learner_encoder.pt").exists() else (ckpt_upstream / "learner_encoder.pt")
        learner_encoder = LearnerStateEncoder(
            embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
            float(cfg["learner_encoder"]["dropout"])).to(device)
        learner_encoder.load_state_dict(torch.load(
            learner_ckpt, map_location=device, weights_only=True))

        time_ckpt = (checkpoint_dir / "time_model.pt") if (checkpoint_dir / "time_model.pt").exists() else (ckpt_upstream / "time_model.pt")
        time_model = PersonalizedTimeCostModel(
            int(cfg["learner_encoder"]["hidden_dim"]), embeddings.shape[1],
            embeddings.shape[0], int(cfg["time_model"]["hidden_dim"]),
            float(cfg["time_model"]["dropout"])).to(device)
        time_state = torch.load(
            time_ckpt, map_location=device, weights_only=True)
        time_model.load_state_dict(time_state, strict=False)
        time_metrics = existing_metrics.get("time_model", {})

    if to_stage <= 4:
        log.info("Reached --to-stage 4. Upstream checkpoints saved to %s.", output)
        metrics = {"time_model": time_metrics}
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    if from_stage <= 5:
        difficulties = concept_difficulties(df, n_concepts)
        weights = concept_weights(df, n_concepts)
        np.save(output / "concept_difficulties.npy", difficulties)
        np.save(output / "concept_weights.npy", weights)

        existing_sim_ckpt = (checkpoint_dir / "cognitive_simulator.pt")
        if not existing_sim_ckpt.exists() and upstream_dir:
            existing_sim_ckpt = (ckpt_upstream / "cognitive_simulator.pt")

        if existing_sim_ckpt.exists() and not cfg.get("simulator", {}).get("retrain", False):
            log.info("Stage 5/6: loading existing pyKT DKT cognitive simulator from %s", existing_sim_ckpt)
            emb_dim = int(cfg.get("simulator", {}).get("emb_size", cfg.get("simulator", {}).get("hidden_dim", 64)))
            simulator = TimeEmittingDKT(n_concepts, emb_dim, float(cfg["data"]["max_elapsed_seconds"])).to(device)
            simulator.load_state_dict(torch.load(existing_sim_ckpt, map_location=device, weights_only=True))
            simulator_metrics = existing_metrics.get("simulator", {"type": "pykt_dkt_trained", "n_concepts": n_concepts})
        else:
            log.info("Stage 5/6: training pyKT DKT cognitive simulator")
            simulator, simulator_metrics = train_simulator(df, cfg, device)
            _save_state(simulator, checkpoint_dir / "cognitive_simulator.pt")
            _save_state(simulator, checkpoint_dir / "dkt_pykt.pt")
    else:
        log.info("Skipping Stage 5 (loading existing simulator / statistics from %s)", upstream_dir)
        difficulties = np.load(_resolve_artifact("concept_difficulties.npy", output, upstream_dir))
        weights = np.load(_resolve_artifact("concept_weights.npy", output, upstream_dir))

        sim_ckpt = (checkpoint_dir / "cognitive_simulator.pt") if (checkpoint_dir / "cognitive_simulator.pt").exists() else (ckpt_upstream / "cognitive_simulator.pt")
        emb_dim = int(cfg.get("simulator", {}).get("emb_size", cfg.get("simulator", {}).get("hidden_dim", 64)))
        simulator = TimeEmittingDKT(
            embeddings.shape[0], emb_dim,
            float(cfg["data"]["max_elapsed_seconds"])).to(device)
        simulator.load_state_dict(torch.load(
            sim_ckpt,
            map_location=device, weights_only=True))
        simulator_metrics = existing_metrics.get("simulator", {})

    if to_stage <= 5:
        log.info("Reached --to-stage 5. Upstream checkpoints and concept priors saved to %s.", output)
        metrics = {
            "time_model": time_metrics,
            "simulator": simulator_metrics,
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    log.info("Stage 6/6: training single-critic adaptive policy")
    train_episodes = policy_episodes(df, cfg, "train")
    decoder, kt_head, policy_history = train_constrained_policy(
        train_episodes, embeddings, prerequisite, prerequisite_confidence,
        difficulties, weights, learner_encoder, time_model, simulator,
        cfg, device, checkpoint_dir=checkpoint_dir)
    _save_state(decoder, checkpoint_dir / "path_decoder.pt")
    _save_state(kt_head, checkpoint_dir / "kt_aux_head.pt")

    evaluation_episodes = policy_episodes(df, cfg, "test")
    if not evaluation_episodes:
        evaluation_episodes = policy_episodes(df, cfg, "val")
    max_eval_episodes = getattr(args, "max_episodes", None)
    if max_eval_episodes is None:
        max_eval_episodes = cfg.get("evaluation", {}).get("max_episodes", None)
    if max_eval_episodes is not None:
        max_eval_episodes = int(max_eval_episodes)
    eval_frac = cfg.get("evaluation", {}).get("budget_fraction", 0.75)
    evaluation = evaluate_policy(
        evaluation_episodes, embeddings=embeddings,
        prerequisite=prerequisite,
        prerequisite_confidence=prerequisite_confidence,
        difficulties=difficulties, concept_weights=weights,
        learner_encoder=learner_encoder, time_model=time_model,
        simulator=simulator, decoder=decoder, cfg=cfg, device=device,
        budget_fraction=eval_frac, max_episodes=max_eval_episodes)
    metrics = {
        "time_model": time_metrics,
        "simulator": simulator_metrics,
        "policy_evaluation": evaluation,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "policy_history.json").write_text(
        json.dumps(policy_history, indent=2), encoding="utf-8")
    log.info("Training complete. Artifacts written to %s", output)


def _load_src_dkt_wrapper(cfg: dict, n_concepts: int):
    """Load unified PyTorch pyKT DKT mastery simulator."""
    from models.src_dkt_mastery_wrapper import SRCDKTMasteryWrapper
    device = cfg.get("device", "cpu")
    max_history = int(cfg.get("data", {}).get("max_history", 100))
    return SRCDKTMasteryWrapper(n_concepts=n_concepts, max_history=max_history, device=device)


def _load_model_bundle(artifact_dir: Path, device: str, upstream_dir: Path | None = None):
    cfg_file = (artifact_dir / "resolved_config.yaml") if (artifact_dir / "resolved_config.yaml").exists() else (artifact_dir / "config.yaml")
    cfg = load_config(cfg_file)
    cfg["device"] = device

    if upstream_dir is None:
        upstream_str = cfg.get("upstream_artifacts")
        upstream_dir = Path(upstream_str).resolve() if upstream_str else artifact_dir.resolve()
    else:
        upstream_dir = Path(upstream_dir).resolve()

    metadata_path = _resolve_artifact("metadata.json", artifact_dir, upstream_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    emb_path = _resolve_artifact("concept_embeddings.npy", artifact_dir, upstream_dir)
    embeddings = torch.tensor(
        np.load(emb_path),
        dtype=torch.float32, device=device)

    graphs_dir = (artifact_dir / "graphs") if (artifact_dir / "graphs").exists() else (upstream_dir / "graphs")
    prerequisite = np.load(graphs_dir / "prerequisite.npy")
    confidence = np.load(graphs_dir / "prerequisite_confidence.npy")

    difficulties = np.load(_resolve_artifact("concept_difficulties.npy", artifact_dir, upstream_dir))
    weights = np.load(_resolve_artifact("concept_weights.npy", artifact_dir, upstream_dir))

    checkpoint_dir = artifact_dir / "checkpoints"
    upstream_ckpt_dir = upstream_dir / "checkpoints"

    learner_ckpt = (checkpoint_dir / "learner_encoder.pt") if (checkpoint_dir / "learner_encoder.pt").exists() else (upstream_ckpt_dir / "learner_encoder.pt")
    learner = LearnerStateEncoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        float(cfg["learner_encoder"]["dropout"])).to(device)
    learner.load_state_dict(torch.load(
        learner_ckpt, map_location=device, weights_only=True))

    time_ckpt = (checkpoint_dir / "time_model.pt") if (checkpoint_dir / "time_model.pt").exists() else (upstream_ckpt_dir / "time_model.pt")
    time_model = PersonalizedTimeCostModel(
        int(cfg["learner_encoder"]["hidden_dim"]), embeddings.shape[1],
        embeddings.shape[0], int(cfg["time_model"]["hidden_dim"]),
        float(cfg["time_model"]["dropout"])).to(device)
    time_state = torch.load(
        time_ckpt, map_location=device, weights_only=True)
    time_model.load_state_dict(time_state, strict=False)

    # Cognitive simulator: PyTorch pyKT DKT simulator
    n_concepts = int(metadata["n_concepts"])
    sim_ckpt = (checkpoint_dir / "cognitive_simulator.pt") if (checkpoint_dir / "cognitive_simulator.pt").exists() else (upstream_ckpt_dir / "cognitive_simulator.pt")
    emb_dim = int(cfg.get("simulator", {}).get("emb_size", cfg.get("simulator", {}).get("hidden_dim", 64)))
    simulator = TimeEmittingDKT(
        embeddings.shape[0], emb_dim,
        float(cfg["data"]["max_elapsed_seconds"])).to(device)
    simulator.load_state_dict(torch.load(
        sim_ckpt,
        map_location=device, weights_only=True))

    zero_budget_features = not cfg.get("ablations", {}).get("use_budget_features", True)
    decoder = BudgetAwarePathDecoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        int(cfg["decoder"]["hidden_dim"]), float(cfg["decoder"]["dropout"]),
        float(cfg["decoder"]["stop_bias"]),
        zero_budget_features=zero_budget_features).to(device)
    decoder_ckpt = (checkpoint_dir / "path_decoder.pt") if (checkpoint_dir / "path_decoder.pt").exists() else (upstream_ckpt_dir / "path_decoder.pt")
    decoder.load_state_dict(torch.load(
        decoder_ckpt, map_location=device, weights_only=True))
    for m in (learner, time_model, decoder):
        m.eval()
    if hasattr(simulator, "eval"):
        simulator.eval()
    return cfg, metadata, embeddings, prerequisite, confidence, difficulties, weights, learner, time_model, simulator, decoder


def _map_concept(value, mapping: dict[str, int]) -> int:
    text = str(value)
    if text in mapping:
        return int(mapping[text])
    index = int(value)
    if index < 0 or index >= len(mapping):
        raise ValueError(f"concept index {index} is outside the trained vocabulary")
    return index


def recommend(args) -> None:
    artifact_dir = Path(args.artifacts)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = _load_model_bundle(artifact_dir, requested_device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner, time_model, simulator, decoder) = bundle
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    mapping = metadata["concept_to_idx"]
    concepts = [_map_concept(value, mapping) for value in payload.get("concepts", [])]
    answers = [int(value) for value in payload.get("correct", [])]
    elapsed = [float(value) for value in payload.get("elapsed_sec", [])]
    lag = [float(value) for value in payload.get("lag_sec", [0.0] * len(concepts))]
    if not (len(concepts) == len(answers) == len(elapsed) == len(lag)):
        raise ValueError("concepts, correct, elapsed_sec, and lag_sec must have equal lengths")
    targets = [_map_concept(value, mapping) for value in payload["targets"]]
    history = {
        "q": np.asarray(concepts, dtype=np.int64),
        "a": np.asarray(answers, dtype=np.int64),
        "t": np.asarray(elapsed, dtype=np.float32),
        "lag": np.asarray(lag, dtype=np.float32),
    }
    planner = AdaptiveTEMPOPlanner(
        embeddings=embeddings, prerequisite=prerequisite,
        prerequisite_confidence=confidence, difficulties=difficulties,
        concept_weights=weights, learner_encoder=learner,
        time_model=time_model, simulator=simulator, decoder=decoder, cfg=cfg,
        history=history, targets=targets,
        budget_seconds=float(payload["budget_seconds"]), device=requested_device)
    result = planner.recommend()
    names = metadata["idx_to_concept"]
    result["path_names"] = [names[index] for index in result["path"]]
    result["next_concept_name"] = (
        None if result["next_concept"] is None else names[result["next_concept"]])
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


def evaluate_command(args) -> None:
    artifact_dir = Path(args.artifacts)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = _load_model_bundle(artifact_dir, requested_device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner, time_model, simulator, decoder) = bundle
    df = pd.read_csv(artifact_dir / "interactions_processed.csv")
    evaluation_episodes = policy_episodes(df, cfg, "test")
    if not evaluation_episodes:
        evaluation_episodes = policy_episodes(df, cfg, "val")

    # Quick Test Mode: Subsample evaluation episodes if specified
    sample_fraction = getattr(args, "sample_fraction", None)
    max_learners = getattr(args, "max_learners", None)
    if sample_fraction is not None and 0.0 < float(sample_fraction) < 1.0:
        n_sub = max(1, int(len(evaluation_episodes) * float(sample_fraction)))
        log.info("Quick Evaluation Mode: Subsampling test episodes by sample_fraction=%s: %d -> %d",
                 sample_fraction, len(evaluation_episodes), n_sub)
        evaluation_episodes = evaluation_episodes[:n_sub]
    elif max_learners is not None and int(max_learners) > 0:
        n_sub = min(len(evaluation_episodes), int(max_learners))
        log.info("Quick Evaluation Mode: Subsampling test episodes to max_learners=%s: %d -> %d",
                 max_learners, len(evaluation_episodes), n_sub)
        evaluation_episodes = evaluation_episodes[:n_sub]
    result = evaluate_policy(
        evaluation_episodes, embeddings=embeddings,
        prerequisite=prerequisite,
        prerequisite_confidence=confidence,
        difficulties=difficulties, concept_weights=weights,
        learner_encoder=learner, time_model=time_model,
        simulator=simulator, decoder=decoder, cfg=cfg,
        device=requested_device,
        budget_fraction=args.budget_fraction,
        max_episodes=args.max_episodes)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive TEMPO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train the pipeline")
    train.add_argument("--data", default=None,
                       help="path to the interaction CSV; defaults to "
                            "<data_dir>/processed/interactions_processed.csv "
                            "as set in config.yaml")
    train.add_argument("--config", default="config.yaml")
    train.add_argument("--prerequisites", help="optional expert prerequisite CSV")
    train.add_argument("--output", default="artifacts")
    train.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                       help="stage to start execution from (1-6); default is 1 (full train)")
    train.add_argument("--to-stage", type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                       help="stage to stop execution after (1-6); default is 6 (full pipeline)")
    train.add_argument("--topology-epochs", type=int, default=None,
                       help="override graph.topology_epochs (e.g. 1000)")
    train.add_argument("--time-epochs", type=int, default=None,
                       help="override time_model.epochs (e.g. 200)")
    train.add_argument("--time-patience", type=int, default=None,
                       help="override time_model.early_stopping_patience (e.g. 25)")
    train.add_argument("--epochs", type=int, default=None, help="override policy decoder epochs")
    train.add_argument("--budget-fractions", nargs="+", type=float, default=None,
                       help="override training.budget_fractions (e.g. --budget-fractions 0.1)")
    train.add_argument("--device", default=None, help="override device (e.g. cuda:0, cuda:1, cpu)")
    train.add_argument("--upstream", default=None,
                       help="path to upstream artifact directory for stages skipped via --from-stage")
    train.add_argument("--sample-fraction", type=float, default=None,
                       help="subsample dataset by learner fraction (e.g. 0.05 for 5%) for quick training")
    train.add_argument("--max-learners", type=int, default=None,
                       help="cap number of unique learners for quick training (e.g. 200)")
    train.set_defaults(function=train_pipeline)

    inference = subparsers.add_parser("recommend", help="generate a safe adaptive plan")
    inference.add_argument("--artifacts", required=True)
    inference.add_argument("--request", required=True, help="learner request JSON")
    inference.add_argument("--device", default="auto")
    inference.add_argument("--output")
    inference.set_defaults(function=recommend)

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a trained policy")
    eval_parser.add_argument("--artifacts", default="artifacts")
    eval_parser.add_argument("--upstream", default=None,
                             help="path to upstream artifact directory for skipped stages")
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--budget-fraction", type=float, default=None)
    eval_parser.add_argument("--max-episodes", type=int, default=None,
                             help="max test episodes to evaluate (default: None for full test set)")
    eval_parser.add_argument("--max-learners", type=int, default=None,
                             help="cap number of test learners to evaluate (e.g. 50)")
    eval_parser.add_argument("--sample-fraction", type=float, default=None,
                             help="fraction of test episodes to evaluate (e.g. 0.05)")
    eval_parser.add_argument("--output")
    eval_parser.set_defaults(function=evaluate_command)

    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)


