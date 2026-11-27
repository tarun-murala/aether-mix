"""Aggregate per-service gate results into JSON + Markdown reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.manifest import Gate, GatesFile
from evals.scoring import GateResult, judge


def _worst(gate: Gate, values: list[float]) -> float:
    if gate.min is not None:
        return min(values)
    return max(values)


def aggregate(results: list[GateResult], gates: GatesFile) -> dict[str, Any]:
    """Gate passes only if it holds in every service (a failure on one voice or drummer is a real failure)."""
    by_gate: dict[str, list[GateResult]] = defaultdict(list)
    for r in results:
        by_gate[r.gate].append(r)
    rows = []
    for name, gate in gates.gates.items():
        rs = by_gate.get(name, [])
        vals = [r.value for r in rs if r.value is not None]
        worst = _worst(gate, vals) if vals else None
        if gate.mode != "headless":
            status = "MANUAL" if gate.mode == "manual" else "MAINSTAGE"
        elif gate.informational:
            status = "INFO"
        elif not vals:
            status = "NOT MEASURED"
        else:
            status = "PASS" if all(r.passed for r in rs if r.value is not None) else "FAIL"
        rows.append({"gate": name, "status": status, "worst": worst, "per_service": {r.service: r.value for r in rs},
                     "description": gate.description, "source": gate.source, "mode": gate.mode})
    scored = [r for r in rows if r["status"] in ("PASS", "FAIL")]
    return {"rows": rows, "passed": sum(r["status"] == "PASS" for r in scored), "scored": len(scored),
            "all_passed": all(r["status"] == "PASS" for r in scored) and bool(scored)}


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return "inf" if v == float("inf") else f"{v:.3g}"


def write_reports(out_dir: Path, results: list[GateResult], gates: GatesFile, meta: dict[str, Any]) -> dict[str, Any]:
    agg = aggregate(results, gates)
    payload = {"meta": meta, "aggregate": agg, "results": [r.__dict__ for r in results]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    services = sorted({r.service for r in results})
    lines = [f"# AetherMix multitrack eval - {meta.get('profile')} profile", ""]
    if meta.get("synthetic"):
        lines += ["> **PROVISIONAL**: these slots used deterministic synthetic stand-ins, not real recordings:", ""]
        for svc, parts in meta["synthetic"].items():
            lines.append(f"> - {svc}: {', '.join(parts) if parts else 'none'}")
        lines.append("")
    lines += [f"Scored gates passed: **{agg['passed']}/{agg['scored']}**", "",
              "| Gate | Status | Worst | " + " | ".join(services) + " | Source |",
              "| --- | --- | --- | " + " | ".join("---" for _ in services) + " | --- |"]
    for r in agg["rows"]:
        lines.append(f"| `{r['gate']}` | {r['status']} | {_fmt(r['worst'])} | "
                     + " | ".join(_fmt(r["per_service"].get(s)) for s in services) + f" | {r['source']} |")
    lines += ["", "## Failures and findings", ""]
    for r in results:
        g = gates.gates[r.gate]
        if g.mode == "headless" and (r.passed is False or (g.informational and r.value)):
            lines.append(f"- **{r.service} / {r.gate}** = {_fmt(r.value)} ({r.detail})")
            for e in r.evidence[:6]:
                lines.append(f"  - {json.dumps(e, default=str)}")
    lines += ["", "## Manual and MainStage gates", ""]
    for name, g in gates.gates.items():
        if g.mode != "headless":
            lines.append(f"- [ ] `{name}` ({g.mode}): {g.description}. {g.metric}. Source: {g.source}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return agg


__all__ = ["aggregate", "judge", "write_reports"]
