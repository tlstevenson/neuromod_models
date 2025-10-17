"""
Unified trainer: select a model and a behavior (task), run training, and save outputs.

Supports:
- Blanco-Pozo model + Two-step task (from model.py)
- Miconi-inspired model + Stimulus-Response/Transitive-Inference-style task (from modelWithMiconiData.py)
- Cross pairings via adapters (from cross_trainers.py):
  - Blanco-Pozo model on Stimulus-Response task
  - Miconi-inspired model on Two-step task

Usage examples:
  python unified_trainer.py --model blanco-pozo --task two-step --episodes 200 --save-dir ../data/exp_bp_two_step
  python unified_trainer.py --model miconi --task stimulus-response --episodes 200 --save-dir ../data/exp_miconi_sr
  python unified_trainer.py --model miconi --task transitive-inference --episodes 200 --save-dir ../data/exp_miconi_ti
  python unified_trainer.py --model blanco-pozo --task bp-on-sr --episodes 200 --save-dir ../data/exp_bp_on_sr
  python unified_trainer.py --model miconi --task miconi-on-two-step --episodes 200 --save-dir ../data/exp_miconi_on_twostep
"""

import os
import sys
import argparse
from typing import Optional, Dict

# Make sure local imports resolve when invoked from repo root or elsewhere
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Import the two pipelines already implemented in this repo
import model as blanco_pozo_mod
import modelWithMiconiData as miconi_mod
# Cross-training adapters and loops
import cross_trainers as cross_mod


def build_params(base_params: dict, overrides: Optional[Dict]) -> dict:
    pm = dict(base_params)
    if overrides:
        for k, v in overrides.items():
            # Ignore None values
            if v is None:
                continue
            pm[k] = v
    return pm


def run(model: str, task: str, episodes: Optional[int], save_dir: Optional[str]):
    model = model.lower()
    task = task.lower()

    if model == "blanco-pozo" and task == "two-step":
        base = blanco_pozo_mod.default_params
        overrides = {}
        if episodes is not None:
            overrides["n_episodes"] = episodes
        params = build_params(base, overrides)
        print(f"Running Blanco-Pozo model on Two-step task for {params['n_episodes']} episodes...")
        if save_dir:
            print(f"Saving to: {save_dir}")
        blanco_pozo_mod.run_simulation(save_dir=save_dir, pm=params)
        return

    if model == "miconi" and task in ("transitive-inference", "ti", "stimulus-response", "sr"):
        base = miconi_mod.default_params
        overrides = {}
        if episodes is not None:
            overrides["n_episodes"] = episodes
        params = build_params(base, overrides)
        print(f"Running Miconi-style model on Stimulus-Response task for {params['n_episodes']} episodes...")
        if save_dir:
            print(f"Saving to: {save_dir}")
        miconi_mod.run_simulation(save_dir=save_dir, pm=params)
        return

    # Cross pairings via adapters
    if model == "blanco-pozo" and task in ("bp-on-sr", "blanco-on-sr", "bp_sr", "sr-with-bp", "stimulus-response-with-bp"):
        # Use Miconi SR defaults as the base, since the task is SR
        base = miconi_mod.default_params
        overrides = {}
        if episodes is not None:
            overrides["n_episodes"] = episodes
        params = build_params(base, overrides)
        print(f"Running Blanco-Pozo model on Stimulus-Response task (adapter) for {params['n_episodes']} episodes...")
        if save_dir:
            print(f"Saving to: {save_dir}")
        cross_mod.run_blanco_on_sr(save_dir=save_dir, pm=params)
        return

    if model == "miconi" and task in ("miconi-on-two-step", "miconi_two_step", "two-step-with-miconi", "two_step_with_miconi"):
        # Use Miconi defaults; Two-step adapter will read its own env defaults if missing
        base = miconi_mod.default_params
        overrides = {}
        if episodes is not None:
            overrides["n_episodes"] = episodes
        params = build_params(base, overrides)
        print(f"Running Miconi-style model on Two-step task (adapter) for {params['n_episodes']} episodes...")
        if save_dir:
            print(f"Saving to: {save_dir}")
        cross_mod.run_miconi_on_two_step(save_dir=save_dir, pm=params)
        return

    raise ValueError(
        f"Unsupported combination model={model}, task={task}.\n"
        "Supported: (blanco-pozo, two-step), (miconi, stimulus-response|transitive-inference), "
        "(blanco-pozo, bp-on-sr), (miconi, miconi-on-two-step)."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Unified model-task trainer")
    p.add_argument("--model", required=True, choices=["blanco-pozo", "miconi"], help="Model to train")
    p.add_argument(
        "--task",
        required=True,
        choices=[
            "two-step",
            "stimulus-response", "sr",
            "transitive-inference", "ti",
            # Cross pairings
            "bp-on-sr", "blanco-on-sr", "bp_sr", "sr-with-bp", "stimulus-response-with-bp",
            "miconi-on-two-step", "miconi_two_step", "two-step-with-miconi", "two_step_with_miconi",
        ],
        help="Behavior/ task (native or cross-adapted)",
    )
    p.add_argument("--episodes", type=int, default=None, help="Number of episodes (overrides defaults)")
    p.add_argument("--save-dir", type=str, default=None, help="Directory to save results (params, episodes, checkpoints)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    run(args.model, args.task, args.episodes, args.save_dir)
