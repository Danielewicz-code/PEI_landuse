"""Config loading and run-folder management.

Every run is driven by ONE yaml file. A yaml can `extends:` another one
(e.g. configs/1968_prince.yaml extends configs/base.yaml), so shared
settings live in one place and each run only lists what is different.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with attribute access: cfg.tiling.tile_size"""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | os.PathLike) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        raw = _deep_merge(dict(base), raw)
    cfg = Config(raw)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: Config) -> None:
    """Expand {root} in every string value of the config (paths, checkpoints, ...)."""
    root = cfg.get("paths", {}).get("root", "")

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, str):
                d[k] = v.replace("{root}", root)

    walk(cfg)


def git_commit(repo_dir: str | os.PathLike | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir or Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir or Path(__file__).resolve().parents[2],
            capture_output=True, text=True,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_dir(cfg: Config) -> Path:
    """Create runs/<run_id>/ and freeze the resolved config + code version in it.

    run_id = <YYYY-MM-DD>_<run.name>. Re-using the same name on the same day
    re-uses the folder, which is what makes stages resumable.
    """
    run = cfg.get("run", {})
    run_id = run.get("id") or f"{_dt.date.today().isoformat()}_{run['name']}"
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("run", {})["id"] = run_id

    frozen = run_dir / "config.yaml"
    if not frozen.exists():
        with open(frozen, "w") as f:
            yaml.safe_dump(dict(cfg), f, sort_keys=False)
        meta = {
            "run_id": run_id,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "operator": run.get("operator", "unknown"),
        }
        with open(run_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    return run_dir


def log(run_dir: Path, msg: str) -> None:
    """Print AND append to runs/<id>/log.txt so Colab output loss doesn't lose history."""
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(Path(run_dir) / "log.txt", "a") as f:
        f.write(line + "\n")
