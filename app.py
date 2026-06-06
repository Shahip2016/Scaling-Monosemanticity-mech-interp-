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

    if args.sae:
        load_checkpoint(args.sae, args.model)

    print(f"\n🧠 Scaling Monosemanticity Dashboard")
    print(f"   → http://{args.host}:{args.port}")
    print(f"   SAE loaded: {'✓' if _state['sae'] else '✗ (demo mode)'}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
