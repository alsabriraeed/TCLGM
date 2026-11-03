"""Fast HTTP API Server for TCLGM Inference & Recommendation."""
from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

from config import load_config
from models.cognitive_simulator import TimeEmittingDKT
from models.learner_state_encoder import LearnerStateEncoder
from models.online_planner import AdaptiveTEMPOPlanner
from models.step2_personalized_time_cost_model import PersonalizedTimeCostModel
from models.step4_budget_aware_path_decoder import BudgetAwarePathDecoder

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Global in-memory cache of loaded bundles
MODEL_BUNDLES = {}
STUDENT_PROFILES = {}

def load_bundle_cached(dataset_name: str, device: str = "cpu"):
    if dataset_name in MODEL_BUNDLES:
        return MODEL_BUNDLES[dataset_name]
    
    art_dir = ARTIFACTS_DIR / dataset_name
    if not art_dir.exists():
        raise FileNotFoundError(f"Dataset artifacts not found at {art_dir}")
        
    cfg = load_config(art_dir / "resolved_config.yaml")
    cfg["device"] = device
    metadata = json.loads((art_dir / "metadata.json").read_text(encoding="utf-8"))
    embeddings = torch.tensor(
        np.load(art_dir / "concept_embeddings.npy"),
        dtype=torch.float32, device=device)
    prerequisite = np.load(art_dir / "graphs" / "prerequisite.npy")
    confidence = np.load(art_dir / "graphs" / "prerequisite_confidence.npy")
    difficulties = np.load(art_dir / "concept_difficulties.npy")
    weights = np.load(art_dir / "concept_weights.npy")
    checkpoint_dir = art_dir / "checkpoints"

    learner = LearnerStateEncoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        float(cfg["learner_encoder"]["dropout"])).to(device)
    learner.load_state_dict(torch.load(
        checkpoint_dir / "learner_encoder.pt", map_location=device, weights_only=True))

    time_model = PersonalizedTimeCostModel(
        int(cfg["learner_encoder"]["hidden_dim"]), embeddings.shape[1],
        embeddings.shape[0], int(cfg["time_model"]["hidden_dim"]),
        float(cfg["time_model"]["dropout"])).to(device)
    time_state = torch.load(
        checkpoint_dir / "time_model.pt", map_location=device, weights_only=True)
    time_model.load_state_dict(time_state, strict=False)

    simulator = TimeEmittingDKT(
        embeddings.shape[0], int(cfg["simulator"]["hidden_dim"]),
        float(cfg["data"]["max_elapsed_seconds"])).to(device)
    simulator.load_state_dict(torch.load(
        checkpoint_dir / "cognitive_simulator.pt", map_location=device, weights_only=True))
        
    zero_budget_features = not cfg.get("ablations", {}).get("use_budget_features", True)
    decoder = BudgetAwarePathDecoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        int(cfg["decoder"]["hidden_dim"]), float(cfg["decoder"]["dropout"]),
        float(cfg["decoder"]["stop_bias"]),
        zero_budget_features=zero_budget_features).to(device)
    decoder.load_state_dict(torch.load(
        checkpoint_dir / "path_decoder.pt", map_location=device, weights_only=True))
        
    for m in (learner, time_model, simulator, decoder):
        m.eval()
        
    bundle = {
        "cfg": cfg,
        "metadata": metadata,
        "embeddings": embeddings,
        "prerequisite": prerequisite,
        "confidence": confidence,
        "difficulties": difficulties,
        "weights": weights,
        "learner": learner,
        "time_model": time_model,
        "simulator": simulator,
        "decoder": decoder,
    }
    MODEL_BUNDLES[dataset_name] = bundle
    print(f"Loaded and cached {dataset_name} bundle successfully.")
    return bundle


def get_student_profiles(dataset_name: str) -> list[dict]:
    if dataset_name in STUDENT_PROFILES:
        return STUDENT_PROFILES[dataset_name]
    
    prof_file = ARTIFACTS_DIR / dataset_name / "student_profiles.json"
    if prof_file.exists():
        profiles = json.loads(prof_file.read_text())
        STUDENT_PROFILES[dataset_name] = profiles
        return profiles
    return []


def run_recommendation(payload: dict) -> dict:
    dataset = payload.get("dataset", "ASSISST2012")
    bundle = load_bundle_cached(dataset)
    cfg = bundle["cfg"]
    metadata = bundle["metadata"]
    embeddings = bundle["embeddings"]
    prerequisite = bundle["prerequisite"]
    confidence = bundle["confidence"]
    difficulties = bundle["difficulties"]
    weights = bundle["weights"]
    learner = bundle["learner"]
    time_model = bundle["time_model"]
    simulator = bundle["simulator"]
    decoder = bundle["decoder"]
    names = metadata["idx_to_concept"]
    mapping = metadata["concept_to_idx"]

    # If uidx is passed, look up student profile
    if "uidx" in payload and "concepts" not in payload:
        uidx = int(payload["uidx"])
        profiles = get_student_profiles(dataset)
        matched = next((p for p in profiles if p["uidx"] == uidx), None)
        if not matched:
            raise ValueError(f"Student #{uidx} not found in {dataset} profiles.")
        raw_concepts = matched["concepts"]
        raw_answers = matched["correct"]
        raw_elapsed = matched["elapsed_sec"]
        raw_lag = matched["lag_sec"]
    else:
        raw_concepts = payload.get("concepts", [])
        raw_answers = payload.get("correct", [])
        raw_elapsed = payload.get("elapsed_sec", [])
        raw_lag = payload.get("lag_sec", [0.0] * len(raw_concepts))

    # Map concept names/ids to int
    def _map(val):
        t = str(val)
        if t in mapping:
            return int(mapping[t])
        idx = int(val)
        if 0 <= idx < len(mapping):
            return idx
        raise ValueError(f"Unknown concept: {val}")

    concepts = [_map(c) for c in raw_concepts]
    answers = [int(a) for a in raw_answers]
    elapsed = [float(e) for e in raw_elapsed]
    lag = [float(l) for l in raw_lag]

    # Resolve budget
    if "budget_minutes" in payload:
        budget_sec = float(payload["budget_minutes"]) * 60.0
    elif "budget_seconds" in payload:
        budget_sec = float(payload["budget_seconds"])
    else:
        budget_sec = 30.0 * 60.0 # Default 30 min

    # Resolve targets (if none passed, pick unmastered frontier concepts)
    if "targets" in payload and payload["targets"]:
        targets = [_map(t) for t in payload["targets"]]
    else:
        # Default targets: pick concepts with lowest mastery
        mastery_est, _ = simulator.mastery_after_history(
            np.asarray(concepts, dtype=np.int64),
            np.asarray(answers, dtype=np.int64))
        # Pick top unmastered concepts
        unmastered = np.argsort(mastery_est)[:3].tolist()
        targets = unmastered

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
        budget_seconds=budget_sec, device="cpu")
        
    result = planner.recommend()
    result["path_names"] = [names[i] for i in result["path"]]
    result["next_concept_name"] = names[result["next_concept"]] if result["next_concept"] is not None else None

    # Diagnostics
    mastery_threshold = float(cfg["goals"]["mastery_threshold"])
    current_mastery, sim_state = simulator.mastery_after_history(
        history["q"], history["a"])

    path = result["path"]
    if path:
        simulation = simulator.simulate_path_src_style(sim_state, path)
        final_mastery = simulation.final_mastery
    else:
        final_mastery = current_mastery.copy()

    def _learning_gain(b: float, a: float) -> float:
        return float((a - b) / (1.0 - b + 1e-9))

    n = len(current_mastery)
    mastery_map = {names[i]: round(float(current_mastery[i]), 4) for i in range(n)}
    predicted_map = {names[i]: round(float(final_mastery[i]), 4) for i in range(n)}

    mastered_before = {k: v for k, v in mastery_map.items() if v >= mastery_threshold}
    mastered_after  = {k: v for k, v in predicted_map.items() if v >= mastery_threshold}
    newly_mastered  = {k: v for k, v in mastered_after.items() if k not in mastered_before}

    current_level   = round(float(np.mean(current_mastery)), 4)
    predicted_level = round(float(np.mean(final_mastery)), 4)
    global_gain     = _learning_gain(current_level, predicted_level)

    target_before = float(np.mean(current_mastery[targets])) if targets else 0.0
    target_after  = float(np.mean(final_mastery[targets])) if targets else 0.0
    target_gain   = _learning_gain(target_before, target_after)

    path_mastery_delta = [
        {
            "step": idx + 1,
            "concept_idx": c,
            "concept": names[c],
            "mastery_before": round(float(current_mastery[c]), 4),
            "mastery_after": round(float(final_mastery[c]), 4),
            "gain": round(_learning_gain(float(current_mastery[c]), float(final_mastery[c])), 4),
        }
        for idx, c in enumerate(path)
    ]

    target_names = [names[t] for t in targets]

    result["diagnostics"] = {
        "dataset": dataset,
        "budget_minutes": round(budget_sec / 60.0, 1),
        "predicted_time_minutes": round(result.get("predicted_time", 0.0) / 60.0, 1),
        "targets": target_names,
        "current_learning_level": current_level,
        "predicted_learning_level_after_path": predicted_level,
        "global_learning_gain": round(global_gain, 4),
        "target_mastery_before": round(target_before, 4),
        "target_mastery_after": round(target_after, 4),
        "target_learning_gain": round(target_gain, 4),
        "n_concepts_mastered_before": len(mastered_before),
        "n_concepts_mastered_after": len(mastered_after),
        "mastered_concepts_before": mastered_before,
        "newly_mastered_after_path": newly_mastered,
        "path_mastery_delta": path_mastery_delta,
        "all_concept_mastery": mastery_map,
    }
    result.pop("predicted_time", None)
    result.pop("remaining_budget", None)
    return result


class RequestHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "datasets": ["ASSISST2012", "ASSISST2017"]}).encode())
            return

        if path == "/api/students":
            dataset = qs.get("dataset", ["ASSISST2012"])[0]
            profiles = get_student_profiles(dataset)
            summary = [
                {
                    "uidx": p["uidx"],
                    "label": p["label"],
                    "n_interactions": p["n_interactions"],
                    "n_unique_concepts": p["n_unique_concepts"],
                    "sample_concepts": p["sample_concepts"],
                }
                for p in profiles
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"students": summary}).encode())
            return

        if path == "/api/concepts":
            dataset = qs.get("dataset", ["ASSISST2012"])[0]
            meta_file = ARTIFACTS_DIR / dataset / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                concepts = [{"idx": i, "name": name} for i, name in enumerate(meta["idx_to_concept"])]
            else:
                concepts = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({"concepts": concepts}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommend":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
                result = run_recommendation(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        self.send_response(404)
        self.end_headers()


def start_server(port: int = 8008):
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"TCLGM API Server running at http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8008
    start_server(port)
