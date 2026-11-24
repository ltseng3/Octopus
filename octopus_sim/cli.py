"""OCTOPUS-Sim command-line interface and durable outputs."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any
from . import __version__
from .config import ExperimentConfig, ConfigurationError
from .metrics import paired_joint_risk_difference
from .simulator import run_simulation, SimulationResult


def _dump_json(path: Path, item: Any) -> None:
    path.write_text(json.dumps(item, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as file:
        for item in records:
            file.write(json.dumps(item, sort_keys=True, separators=(',', ':')) + '\n')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save(result: SimulationResult, dest: Path, trace: Path | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _dump_json(dest / 'summary.json', result.summary)
    _dump_json(dest / 'scenario_resolved.json', result.config)
    _write_jsonl(dest / 'scores.jsonl', result.scores)
    _write_jsonl(dest / 'inferences.jsonl', result.inferences)
    _write_jsonl(dest / 'events.jsonl', result.events)
    _write_jsonl(dest / 'plans.jsonl', result.policy_plans)
    _write_jsonl(dest / 'operations.jsonl', result.operations or [])
    if result.candidate_estimates is not None:
        _write_jsonl(dest / 'candidates.jsonl', result.candidate_estimates)
    if result.fusion_decisions is not None:
        _write_jsonl(dest / 'fusion_decisions.jsonl', result.fusion_decisions)
    config_digest = hashlib.sha256(json.dumps(result.config, sort_keys=True).encode()).hexdigest()
    manifest = {
        'octopus_version': __version__, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'config_sha256': config_digest,
        'backend': result.config['world']['backend'],
        'scheduler': 'stdlib_heapq_fifo_v1',
        'execution_granularity': result.config['world']['execution_granularity'],
        'energy_model': ('device_interval_ledger_v1' if result.config['energy']['enabled'] else None),
        'energy_calibration': (result.config['energy']['calibration'] if result.config['energy']['enabled'] else None),
        'trace_sha256': _sha256(trace) if trace is not None else None,
        'coordination_contract': ('policy_observation_v1' if result.config['coordination']['enabled'] else None),
    }
    _dump_json(dest / 'manifest.json', manifest)


def _run(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.load(args.config)
    result = run_simulation(cfg, detailed_events=not args.no_events)
    save(result, Path(args.output), cfg.resolve_trace() if cfg.data['world']['backend'] == 'trace' else None)
    print(f"{result.summary['scenario']} | {result.summary['policy']} | "
          f"J_all={result.summary['joint_all_error_rate']:.5f} | "
          f"edge_jobs={result.summary['edge_inference_executions']} | "
          f"deadline_miss={result.summary['agent_deadline_miss_rate']:.4f}")
    print(f'Wrote {args.output}')


def _compare(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.load(args.config)
    policies = [x.strip() for x in args.policies.split(',') if x.strip()]
    variants: list[tuple[str, ExperimentConfig]] = []
    for name in policies:
        if name in {'randomized_reuse', 'mixed_local_shared', 'mixed_local_fusion'}:
            for p_text in args.reuse_grid.split(','):
                p = float(p_text.strip())
                variants.append((f'{name}_{p:g}', cfg.with_policy(name, reuse_probability=p)))
        elif name == 'risk_budget':
            for risk in args.risk_grid.split(','):
                b = float(risk.strip())
                p = cfg.data['policy']
                variants.append((f'risk_budget_{b:g}', cfg.with_policy(name, joint_failure_budget=b,
                    assumed_edge_error=p.get('assumed_edge_error', 0.06),
                    assumed_edge_correlation=p.get('assumed_edge_correlation', 0.0))))
        else:
            variants.append((name, cfg.with_policy(name)))
    if len(set(name for name, _ in variants)) != len(variants):
        raise ValueError('Duplicate policy variant names')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}
    for name, config in variants:
        result = run_simulation(config, detailed_events=not args.no_events)
        save(result, out / name, config.resolve_trace() if config.data['world']['backend'] == 'trace' else None)
        results[name] = result
        rows.append({'variant': name, **result.summary})
        print(f'{name:25s} J_all={result.summary["joint_all_error_rate"]:.5f} '
              f'edge_jobs={result.summary["edge_inference_executions"]:6d} '
              f'complete={result.summary["complete_event_rate"]:.3f}')
    with (out / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = list(results)[0]
    contrasts = {name: paired_joint_risk_difference(results[name].scores, results[baseline].scores)
                 for name in results if name != baseline}
    _dump_json(out / 'paired_comparisons_vs_first.json',
               {'baseline': baseline, 'contrasts': contrasts})
    print(f'Wrote {out / "comparison.csv"}')


def _validate(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    # Loading the world also validates trace candidate completeness.
    from .world import World
    world = World(config.data, config.resolve_trace() if config.data['world']['backend'] == 'trace' else None)
    print(f'VALID: {config.data["scenario"]["name"]}; '
          f'{len(world.events)} events, {len(config.data["scenario"]["agents"])} agents')


def _import_boxes(args: argparse.Namespace) -> None:
    from .datasets.opv2v import import_box_manifest
    offsets = tuple((float(pair.split(',')[0]), float(pair.split(',')[1]))
                    for pair in args.offsets.split(';'))
    result = import_box_manifest(
        args.manifest, args.output, radius_m=args.radius,
        offsets_m=offsets, max_common_range_m=args.common_range,
        forbid_prediction_alias=not args.allow_identical_candidate_files,
    )
    print(f'Converted {result.source_frames} synchronized frames into '
          f'{result.valid_events} common-decision events '
          f'({result.positive_events} positive, {result.negative_events} negative)')
    print(f'Wrote {result.trace_path}; provenance: {result.provenance_path}')


def _empirical_analyze(args: argparse.Namespace) -> None:
    from .datasets.analysis import analyze_empirical
    out = analyze_empirical(args.comparison, args.provenance, baseline=args.baseline,
                            iterations=args.iterations, seed=args.seed, output_file=args.output)
    print(f"Analyzed {out['event_count']} events across {out['scene_count']} scenes")
    if 'warning' in out:
        print('WARNING: '+out['warning'])
    print(f'Wrote {args.output}')


def _plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting needs matplotlib (optional dependency)') from exc
    path = Path(args.comparison_csv)
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        x = float(row['edge_inference_executions']) / float(row['episodes'])
        y = float(row['joint_all_error_rate'])
        ax.scatter(x, y, s=65)
        ax.annotate(row['variant'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Mean edge inference executions per decision event')
    ax.set_ylabel('P(all agents wrong)')
    ax.set_title('Joint-risk vs. compute tradeoff')
    ax.grid(alpha=.2)
    fig.tight_layout()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=175)
    plt.close(fig)
    print(f'Wrote {dest}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='octopus', description='Edge coordinated inference benchmark')
    commands = parser.add_subparsers(dest='command', required=True)
    p_validate = commands.add_parser('validate', help='Validate scenario and trace')
    p_validate.add_argument('config')
    p_validate.set_defaults(func=_validate)
    p_run = commands.add_parser('run', help='Run a scenario')
    p_run.add_argument('config')
    p_run.add_argument('--output', default='outputs/single')
    p_run.add_argument('--no-events', action='store_true', help='Skip detailed event log')
    p_run.set_defaults(func=_run)
    p_compare = commands.add_parser('compare', help='Paired multi-policy experiments')
    p_compare.add_argument('config')
    p_compare.add_argument('--output', default='outputs/comparison')
    p_compare.add_argument('--policies', default='local_only,always_independent,always_share,randomized_reuse,risk_budget')
    p_compare.add_argument('--reuse-grid', default='0,0.1,0.25,0.5,0.75,1')
    p_compare.add_argument('--risk-grid', default='0.01')
    p_compare.add_argument('--no-events', action='store_true')
    p_compare.set_defaults(func=_compare)
    p_import = commands.add_parser('import-opv2v', help='Import synchronized OPV2V/V2V4Real model prediction boxes')
    p_import.add_argument('manifest', help='JSONL frame manifest with actual prediction files')
    p_import.add_argument('--output', default='outputs/empirical_trace')
    p_import.add_argument('--radius', type=float, default=5.0)
    p_import.add_argument('--common-range', type=float, default=45.0)
    p_import.add_argument('--offsets', default='-12,0;0,0;12,0',
                          help='Semicolon-separated XY meter offsets relative to mean agent position')
    p_import.add_argument('--allow-identical-candidate-files', action='store_true',
                          help='Allow identical file paths for local and edge; not independent evidence')
    p_import.set_defaults(func=_import_boxes)
    p_analysis = commands.add_parser('analyze-empirical', help='Scene-clustered paired analysis of real-trace policy runs')
    p_analysis.add_argument('comparison', help='Comparison result directory with per-policy scores.jsonl')
    p_analysis.add_argument('provenance', help='Provenance.jsonl created during importing')
    p_analysis.add_argument('--output', default='outputs/empirical_analysis.json')
    p_analysis.add_argument('--baseline')
    p_analysis.add_argument('--iterations', type=int, default=2000)
    p_analysis.add_argument('--seed', type=int, default=42)
    p_analysis.set_defaults(func=_empirical_analyze)
    p_plot = commands.add_parser('plot', help='Plot joint-risk vs edge executions')
    p_plot.add_argument('comparison_csv')
    p_plot.add_argument('--output', default='outputs/comparison/risk_vs_compute.png')
    p_plot.set_defaults(func=_plot)
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (ConfigurationError, ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
