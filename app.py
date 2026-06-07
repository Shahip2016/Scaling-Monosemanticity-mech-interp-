"""Flask server for the Scaling Monosemanticity interactive dashboard.

Serves the static frontend and exposes REST API endpoints for
feature analysis, steering, and training data.

Usage:
    python app.py                     # Start on port 5000
    python app.py --port 8080         # Custom port
    python app.py --sae checkpoints/sae/sae_final.pt   # Load SAE checkpoint
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")

# Global state — populated on startup if checkpoints exist
_state: dict = {
    "sae": None,
    "model": None,
    "activations": None,
    "training_summary": None,
    "scaling_results": None,
}


def check_or_create_activations(d_model=768, num_vectors=20000, path="data/activations.pt"):
    import torch
    p = Path(path)
    if p.exists():
        return
    
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating synthetic activation data for GPT-2 ({num_vectors} vectors, dimension {d_model})...")
    # Synthetic activations: normal distribution scaled to simulate residual stream
    activations = torch.randn(num_vectors, d_model)
    # Normalize activations
    from src.scaling_monosemanticity.utils import normalize_activations
    activations = normalize_activations(activations, d_model)
    
    torch.save({
        "activations": activations,
        "hook_name": f"blocks.6.hook_resid_post",
        "model_name": "gpt2",
        "d_model": d_model
    }, str(p))
    print(f"Saved synthetic activations to {p}")


class SAETrainingManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle" # idle, training, completed, failed, stopped
        self.progress = 0
        self.total_steps = 0
        self.history = []
        self.logs = []
        self.config = {}
        self.elapsed_s = 0.0
        self.error = None
        self.stop_event = None
        self._thread = None

    def start_training(self, train_config, activations):
        with self.lock:
            if self.status == "training":
                return False
            self.status = "training"
            self.progress = 0
            self.total_steps = train_config.n_steps
            self.history = []
            self.logs = [f"Starting training on {train_config.device}..."]
            self.config = {
                "n_features": train_config.n_features,
                "l1_coefficient": train_config.l1_coefficient,
                "learning_rate": train_config.learning_rate,
                "batch_size": train_config.batch_size,
                "n_steps": train_config.n_steps,
                "device": train_config.device,
            }
            self.elapsed_s = 0.0
            self.error = None
            self.stop_event = threading.Event()
            
            self._thread = threading.Thread(
                target=self._run_training_loop,
                args=(activations, train_config),
                daemon=True
            )
            self._thread.start()
            return True

    def stop_training(self):
        with self.lock:
            if self.status != "training":
                return False
            if self.stop_event:
                self.stop_event.set()
            self.status = "stopped"
            self.logs.append("Stop signal sent. Terminating training...")
            return True

    def _run_training_loop(self, activations, train_config):
        from src.scaling_monosemanticity.train import train_sae
        start_time = time.time()

        def progress_cb(step, record, summary=None):
            with self.lock:
                self.progress = step
                self.elapsed_s = time.time() - start_time
                if record:
                    if not self.history or self.history[-1]["step"] != step:
                        self.history.append(record)
                    log_line = f"step {step:6d} | loss {record['loss']:.4f} | mse {record['mse']:.4f} | l0 {record['l0']:.1f} | var {record['variance_explained']:.3f}"
                    self.logs.append(log_line)
                if summary:
                    self.status = "completed"
                    self.logs.append(f"Training completed successfully! Saved final model to {summary.get('checkpoint')}")
                    # Hot-reload
                    try:
                        self.logs.append("Hot-reloading newly trained model...")
                        _state["training_summary"] = summary
                        load_checkpoint(summary.get("checkpoint"), "gpt2")
                    except Exception as e:
                        self.logs.append(f"Error during hot-reload: {e}")

        try:
            train_sae(
                activations,
                train_config,
                progress_callback=progress_cb,
                stop_event=self.stop_event
            )
        except Exception as e:
            with self.lock:
                self.status = "failed"
                self.error = str(e)
                self.logs.append(f"Training failed: {e}")

    def get_status(self):
        with self.lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "total_steps": self.total_steps,
                "history": list(self.history),
                "logs": list(self.logs),
                "config": dict(self.config),
                "elapsed_s": self.elapsed_s,
                "error": self.error
            }


_training_manager = SAETrainingManager()



# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


# ---------------------------------------------------------------------------
# API: Health check
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "sae_loaded": _state["sae"] is not None,
        "model_loaded": _state["model"] is not None,
        "activations_loaded": _state["activations"] is not None,
    })


# ---------------------------------------------------------------------------
# API: Feature analysis
# ---------------------------------------------------------------------------

@app.route("/api/features/<int:feature_idx>")
def api_feature(feature_idx: int):
    sae = _state["sae"]
    activations = _state["activations"]

    if sae is None or activations is None:
        return jsonify({"error": "No SAE or activations loaded. Using demo data."}), 404

    try:
        from src.scaling_monosemanticity.analysis import analyze_feature
        result = analyze_feature(sae, activations, feature_idx)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features/search")
def api_feature_search():
    keyword = request.args.get("keyword", "")
    if not keyword:
        return jsonify({"error": "keyword parameter required"}), 400

    sae = _state["sae"]
    model = _state["model"]
    activations = _state["activations"]

    if sae is None or model is None:
        return jsonify({"error": "No SAE or model loaded. Using demo data."}), 404

    try:
        from src.scaling_monosemanticity.analysis import find_features_by_keyword
        results = find_features_by_keyword(
            sae, model, activations, [], keyword, top_k_features=10
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Steering
# ---------------------------------------------------------------------------

@app.route("/api/steering", methods=["POST"])
def api_steering():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    feature_idx = data.get("feature_idx", 0)
    coefficient = data.get("coefficient", 10.0)

    sae = _state["sae"]
    model = _state["model"]

    if sae is None or model is None:
        return jsonify({"error": "No SAE or model loaded. Using demo data."}), 404

    try:
        from src.scaling_monosemanticity.steering import compare_steering
        results = compare_steering(
            model, sae, prompt, feature_idx,
            coefficients=[float(coefficient)],
            max_new_tokens=40,
        )
        baseline = results.get("baseline", "")
        steered_key = f"coef_{float(coefficient)}"
        steered = results.get(steered_key, baseline)
        return jsonify({"baseline": baseline, "steered": steered})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Scaling laws
# ---------------------------------------------------------------------------

@app.route("/api/scaling-laws")
def api_scaling_laws():
    if _state["scaling_results"]:
        return jsonify(_state["scaling_results"])

    # Try to load from disk
    candidates = [
        Path("results/scaling_laws/sweep_results.json"),
        Path("scaling_laws/sweep_results.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            _state["scaling_results"] = data
            return jsonify(data)

    return jsonify({"error": "No scaling law data found. Using demo data."}), 404


# ---------------------------------------------------------------------------
# API: Training summary
# ---------------------------------------------------------------------------

@app.route("/api/training-summary")
def api_training_summary():
    if _state["training_summary"]:
        return jsonify(_state["training_summary"])

    # Try to load from disk
    candidates = [
        Path("checkpoints/sae/train_summary.json"),
        Path("checkpoints/train_summary.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            _state["training_summary"] = data
            return jsonify(data)

    return jsonify({"error": "No training summary found. Using demo data."}), 404


# ---------------------------------------------------------------------------
# API: Custom SAE Training
# ---------------------------------------------------------------------------

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    data = request.get_json(force=True) or {}
    
    n_features = int(data.get("n_features", 4096))
    l1_coefficient = float(data.get("l1_coefficient", 5.0))
    learning_rate = float(data.get("learning_rate", 3e-4))
    batch_size = int(data.get("batch_size", 4096))
    n_steps = int(data.get("n_steps", 1000))
    log_every = int(data.get("log_every", 100))
    save_every = int(data.get("save_every", 1000))
    device = data.get("device", "auto")

    # Determine activation dimensions
    d_in = 768
    if _state["model"] is not None:
        try:
            d_in = _state["model"].cfg.d_model
        except Exception:
            pass

    # Verify/load activations
    activations_path = "data/activations.pt"
    if _state["activations"] is None:
        if not Path(activations_path).exists():
            check_or_create_activations(d_model=d_in)
        
        try:
            from src.scaling_monosemanticity.utils import load_activations
            _state["activations"] = load_activations(activations_path)
            print(f"✓ Loaded activations from {activations_path}")
        except Exception as e:
            return jsonify({"error": f"Failed to load activations: {str(e)}"}), 500

    activations = _state["activations"]
    if activations.shape[-1] != d_in:
        print(f"Activation dimension mismatch ({activations.shape[-1]} vs model's {d_in}). Recreating...")
        try:
            if Path(activations_path).exists():
                Path(activations_path).unlink()
        except Exception as e:
            print(f"Failed to delete mismatched activations: {e}")
        check_or_create_activations(d_model=d_in)
        from src.scaling_monosemanticity.utils import load_activations
        _state["activations"] = load_activations(activations_path)
        activations = _state["activations"]

    import torch
    from src.scaling_monosemanticity.train import TrainConfig
    
    device_val = device
    if device_val == "auto" or not device_val:
        device_val = "cuda" if torch.cuda.is_available() else "cpu"
        
    config = TrainConfig(
        n_features=n_features,
        d_in=d_in,
        l1_coefficient=l1_coefficient,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        log_every=log_every,
        save_every=save_every,
        device=device_val,
    )

    success = _training_manager.start_training(config, activations)
    if success:
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "Training is already running"}), 400


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    success = _training_manager.stop_training()
    if success:
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"error": "Training is not running"}), 400


@app.route("/api/train/status")
def api_train_status():
    return jsonify(_training_manager.get_status())


# ---------------------------------------------------------------------------
# Startup: optionally load SAE and model
# ---------------------------------------------------------------------------

def load_checkpoint(sae_path: str | None, model_name: str = "gpt2") -> None:
    """Load SAE checkpoint and optionally the transformer model."""
    if sae_path and Path(sae_path).exists():
        try:
            import torch
            from src.scaling_monosemanticity.sae import SparseAutoencoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sae = SparseAutoencoder.load(sae_path, device=device)
            _state["sae"] = sae
            print(f"✓ Loaded SAE from {sae_path}")

            # Try loading model
            try:
                from transformer_lens import HookedTransformer
                model = HookedTransformer.from_pretrained(
                    model_name, device=device, center_writing_weights=False
                )
                _state["model"] = model
                print(f"✓ Loaded model: {model_name}")
            except ImportError:
                print("⚠ TransformerLens not installed — model features disabled")

        except Exception as e:
            print(f"⚠ Could not load SAE: {e}")

    # Load activations if file exists and not loaded yet
    activations_path = "data/activations.pt"
    if Path(activations_path).exists() and _state["activations"] is None:
        try:
            from src.scaling_monosemanticity.utils import load_activations
            _state["activations"] = load_activations(activations_path)
            print(f"✓ Loaded activations from {activations_path}")
        except Exception as e:
            print(f"⚠ Could not load activations: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scaling Monosemanticity Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--sae", type=str, default=None, help="Path to SAE checkpoint")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load default model/activations/SAE if they exist
    if args.sae:
        load_checkpoint(args.sae, args.model)
    else:
        load_checkpoint("checkpoints/sae/sae_final.pt", args.model)

    print(f"\n🧠 Scaling Monosemanticity Dashboard")
    print(f"   → http://{args.host}:{args.port}")
    print(f"   SAE loaded: {'✓' if _state['sae'] else '✗ (demo mode)'}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
