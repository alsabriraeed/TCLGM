"""Fast Unified Web Server for TCLGM UI and Recommendation Inference."""
from __future__ import annotations

import json
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
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

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
WEB_DIR = BASE_DIR / "web"

MODEL_BUNDLES = {}
STUDENT_PROFILES = {}

def get_bundle(dataset: str = "ASSISST2012", device: str = "cpu"):
    if dataset in MODEL_BUNDLES:
        return MODEL_BUNDLES[dataset]
    
    art = ARTIFACTS_DIR / dataset
    if not art.exists():
        raise FileNotFoundError(f"Artifacts for {dataset} not found at {art}")
        
    cfg = load_config(art / "resolved_config.yaml")
    cfg["device"] = device
    metadata = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
    embeddings = torch.tensor(np.load(art / "concept_embeddings.npy"), dtype=torch.float32, device=device)
    prereq = np.load(art / "graphs" / "prerequisite.npy")
    confidence = np.load(art / "graphs" / "prerequisite_confidence.npy")
    difficulties = np.load(art / "concept_difficulties.npy")
    weights = np.load(art / "concept_weights.npy")
    ckpt = art / "checkpoints"

    learner = LearnerStateEncoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        float(cfg["learner_encoder"]["dropout"])).to(device)
    learner.load_state_dict(torch.load(ckpt / "learner_encoder.pt", map_location=device, weights_only=True))

    time_model = PersonalizedTimeCostModel(
        int(cfg["learner_encoder"]["hidden_dim"]), embeddings.shape[1],
        embeddings.shape[0], int(cfg["time_model"]["hidden_dim"]),
        float(cfg["time_model"]["dropout"])).to(device)
    time_model.load_state_dict(torch.load(ckpt / "time_model.pt", map_location=device, weights_only=True), strict=False)

    simulator = TimeEmittingDKT(
        embeddings.shape[0], int(cfg["simulator"]["hidden_dim"]),
        float(cfg["data"]["max_elapsed_seconds"])).to(device)
    simulator.load_state_dict(torch.load(ckpt / "cognitive_simulator.pt", map_location=device, weights_only=True))

    decoder = BudgetAwarePathDecoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        int(cfg["decoder"]["hidden_dim"]), float(cfg["decoder"]["dropout"]),
        float(cfg["decoder"]["stop_bias"])).to(device)
    decoder.load_state_dict(torch.load(ckpt / "path_decoder.pt", map_location=device, weights_only=True))

    for m in (learner, time_model, simulator, decoder):
        m.eval()

    bundle = {
        "cfg": cfg, "metadata": metadata, "embeddings": embeddings,
        "prerequisite": prereq, "confidence": confidence,
        "difficulties": difficulties, "weights": weights,
        "learner": learner, "time_model": time_model,
        "simulator": simulator, "decoder": decoder,
    }
    MODEL_BUNDLES[dataset] = bundle
    return bundle


def get_students(dataset: str = "ASSISST2012"):
    if dataset in STUDENT_PROFILES:
        return STUDENT_PROFILES[dataset]
    prof_file = ARTIFACTS_DIR / dataset / "student_profiles.json"
    if prof_file.exists():
        data = json.loads(prof_file.read_text(encoding="utf-8"))
        STUDENT_PROFILES[dataset] = data
        return data
    return []


def run_inference(payload: dict):
    dataset = payload.get("dataset", "ASSISST2012")
    bundle = get_bundle(dataset)
    cfg = bundle["cfg"]
    meta = bundle["metadata"]
    names = meta["idx_to_concept"]
    mapping = meta["concept_to_idx"]

    # Student history lookup
    if "uidx" in payload and "concepts" not in payload:
        uidx = int(payload["uidx"])
        profiles = get_students(dataset)
        matched = next((p for p in profiles if p["uidx"] == uidx), None)
        if not matched:
            raise ValueError(f"Student #{uidx} not found in {dataset}")
        raw_c, raw_a, raw_e, raw_l = matched["concepts"], matched["correct"], matched["elapsed_sec"], matched["lag_sec"]
    else:
        raw_c = payload.get("concepts", [])
        raw_a = payload.get("correct", [])
        raw_e = payload.get("elapsed_sec", [])
        raw_l = payload.get("lag_sec", [0.0] * len(raw_c))

    def _map(val):
        t = str(val)
        if t in mapping:
            return int(mapping[t])
        idx = int(val)
        if 0 <= idx < len(mapping):
            return idx
        raise ValueError(f"Unknown concept: {val}")

    concepts = [_map(c) for c in raw_c]
    answers = [int(a) for a in raw_a]
    elapsed = [float(e) for e in raw_e]
    lag = [float(l) for l in raw_l]

    # Resolve minutes
    if "budget_minutes" in payload:
        budget_sec = float(payload["budget_minutes"]) * 60.0
    elif "budget_seconds" in payload:
        budget_sec = float(payload["budget_seconds"])
    else:
        budget_sec = 30.0 * 60.0

    # Simulator estimate before path
    mastery_before, sim_state = bundle["simulator"].mastery_after_history(
        np.asarray(concepts, dtype=np.int64),
        np.asarray(answers, dtype=np.int64))

    # Resolve targets
    if "targets" in payload and payload["targets"]:
        targets = [_map(t) for t in payload["targets"]]
    else:
        # Default: pick top unmastered frontier concepts with lowest mastery
        unmastered = np.argsort(mastery_before)[:4].tolist()
        targets = unmastered

    history = {
        "q": np.asarray(concepts, dtype=np.int64),
        "a": np.asarray(answers, dtype=np.int64),
        "t": np.asarray(elapsed, dtype=np.float32),
        "lag": np.asarray(lag, dtype=np.float32),
    }

    # Override max_steps to the full number of concepts in the dataset
    # so the path can extend up to the entire concept space (not capped at 25).
    import copy as _copy
    cfg_runtime = _copy.deepcopy(cfg)
    cfg_runtime["decoder"]["max_steps"] = len(names)

    planner = AdaptiveTEMPOPlanner(
        embeddings=bundle["embeddings"],
        prerequisite=bundle["prerequisite"],
        prerequisite_confidence=bundle["confidence"],
        difficulties=bundle["difficulties"],
        concept_weights=bundle["weights"],
        learner_encoder=bundle["learner"],
        time_model=bundle["time_model"],
        simulator=bundle["simulator"],
        decoder=bundle["decoder"],
        cfg=cfg_runtime, history=history, targets=targets,
        budget_seconds=budget_sec, device="cpu")

    result = planner.recommend()
    path = result["path"]
    result["path_names"] = [names[i] for i in path]
    result["next_concept_name"] = names[result["next_concept"]] if result["next_concept"] is not None else None

    # Simulate path completion
    if path:
        simulation = bundle["simulator"].simulate_path_src_style(sim_state, path)
        final_mastery = simulation.final_mastery
    else:
        final_mastery = mastery_before.copy()

    def _gain(b: float, a: float) -> float:
        return float((a - b) / (1.0 - b + 1e-9))

    threshold = float(cfg["goals"]["mastery_threshold"])
    n = len(mastery_before)

    # Mastered sets
    # Only concepts explicitly in the recommended path can be newly mastered.
    # The DKT LSTM propagates hidden state across all concepts, causing phantom
    # mastery rises in unrelated concepts. We restrict newly_mastered and
    # n_concepts_mastered_after to: (a) concepts already mastered before the
    # path, plus (b) path concepts whose final_mastery >= threshold.
    path_set = set(path)
    mastered_before_dict = {}
    mastered_after_dict = {}
    for i in range(n):
        if mastery_before[i] >= threshold:
            mastered_before_dict[names[i]] = round(float(mastery_before[i]), 4)
        # Count a concept as mastered-after only if it was already mastered,
        # or it is in the recommended path and crossed the threshold.
        if final_mastery[i] >= threshold and (i in path_set or mastery_before[i] >= threshold):
            mastered_after_dict[names[i]] = round(float(final_mastery[i]), 4)

    newly_mastered = {k: v for k, v in mastered_after_dict.items() if k not in mastered_before_dict}

    current_level = round(float(np.mean(mastery_before)), 4)
    predicted_level = round(float(np.mean(final_mastery)), 4)
    global_gain = _gain(current_level, predicted_level)

    target_b = float(np.mean(mastery_before[targets])) if targets else 0.0
    target_a = float(np.mean(final_mastery[targets])) if targets else 0.0
    target_gain = _gain(target_b, target_a)

    # Per-concept step diagnostics with prerequisite info
    path_steps = []
    for idx, c in enumerate(path):
        # Find explicit prerequisites for this concept
        prereqs = np.flatnonzero(bundle["prerequisite"][:, c]).tolist()
        prereq_names = [names[p] for p in prereqs]
        path_steps.append({
            "step": idx + 1,
            "concept_idx": c,
            "concept_name": names[c],
            "difficulty": round(float(bundle["difficulties"][c]), 3),
            "mastery_before": round(float(mastery_before[c]), 4),
            "mastery_after": round(float(final_mastery[c]), 4),
            "gain": round(_gain(float(mastery_before[c]), float(final_mastery[c])), 4),
            "prerequisites": prereq_names,
            "is_target": c in targets,
        })

    # Mastered concepts formatted list with difficulty
    mastered_list = [
        {
            "concept_idx": i,
            "concept_name": names[i],
            "mastery": round(float(mastery_before[i]), 4),
            "difficulty": round(float(bundle["difficulties"][i]), 3),
        }
        for i in range(n) if mastery_before[i] >= threshold
    ]
    mastered_list.sort(key=lambda x: -x["mastery"])

    # Targets formatted
    target_list = [
        {
            "concept_idx": t,
            "concept_name": names[t],
            "mastery_before": round(float(mastery_before[t]), 4),
            "mastery_after": round(float(final_mastery[t]), 4),
            "gain": round(_gain(float(mastery_before[t]), float(final_mastery[t])), 4),
        }
        for t in targets
    ]

    result["diagnostics"] = {
        "dataset": dataset,
        "budget_minutes": round(budget_sec / 60.0, 1),
        "predicted_time_minutes": round(result.get("predicted_time", 0.0) / 60.0, 1),
        "current_learning_level": current_level,
        "predicted_learning_level_after_path": predicted_level,
        "global_learning_gain": round(global_gain, 4),
        "target_mastery_before": round(target_b, 4),
        "target_mastery_after": round(target_a, 4),
        "target_learning_gain": round(target_gain, 4),
        "n_concepts_mastered_before": len(mastered_before_dict),
        "n_concepts_mastered_after": len(mastered_after_dict),
        "mastered_concepts": mastered_list,
        "newly_mastered": newly_mastered,
        "path_steps": path_steps,
        "targets": target_list,
        "frontier_names": [names[f] for f in result.get("frontier", []) if f < len(names)],
    }
    result.pop("predicted_time", None)
    result.pop("remaining_budget", None)
    return result


class WebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "datasets": ["ASSISST2012", "ASSISST2017"]}).encode())
            return

        if path == "/api/students":
            dataset = qs.get("dataset", ["ASSISST2012"])[0]
            profiles = get_students(dataset)
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
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"students": summary}).encode())
            return

        if path == "/api/concepts":
            dataset = qs.get("dataset", ["ASSISST2012"])[0]
            ordered_file = Path(f"/home/raed/TCLGM/artifacts/{dataset}/concepts_ordered.json")
            if ordered_file.exists():
                concepts = json.loads(ordered_file.read_text())
            else:
                bundle = get_bundle(dataset)
                meta = bundle["metadata"]
                concepts = [
                    {"idx": i, "name": name, "difficulty": round(float(bundle["difficulties"][i]), 3)}
                    for i, name in enumerate(meta["idx_to_concept"])
                ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"concepts": concepts}).encode())
            return

        if path == "/api/graph-data":
            dataset = qs.get("dataset", ["ASSISST2012"])[0]
            bundle = get_bundle(dataset)
            meta = bundle["metadata"]
            names = meta["idx_to_concept"]
            prereq = bundle["prerequisite"]
            conf = bundle["confidence"]
            diff = bundle["difficulties"]

            nodes = [
                {"id": i, "name": names[i], "difficulty": round(float(diff[i]), 3)}
                for i in range(len(names))
            ]
            edges = []
            rows, cols = np.nonzero(prereq)
            for s, t in zip(rows, cols):
                edges.append({"source": int(s), "target": int(t), "confidence": round(float(conf[s, t]), 3)})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"nodes": nodes, "edges": edges}).encode())
            return

        # Default static file serving from web/
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/recommend":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
                result = run_inference(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        self.send_response(404)
        self.end_headers()


def run_web(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), WebHandler)
    print(f"🚀 TCLGM Web Dashboard running at http://localhost:{port}")
    print(f"👉 Path Recommendation & Graph Visualizer ready!")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_web(port)
