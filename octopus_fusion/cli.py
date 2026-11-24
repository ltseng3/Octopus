"""Standard-library CLI: toy cases, external JSONL, output comparison/decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .benchmark import compare, save_comparison, save_decisions
from .contracts import FusionConfig
from .estimators import QualityProxyEstimator
from .io import load_labeled_cases, write_labeled_cases
from .synthetic import ToySettings, generate


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(prog='octopus-fusion',description='OCTOPUS F0 offline selective-fusion baseline')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('toy',help='generate and benchmark deterministic synthetic frame/ROI decisions')
    p.add_argument('--frames',type=int,default=1800)
    p.add_argument('--agents',type=int,default=3)
    p.add_argument('--rois',type=int,default=2)
    p.add_argument('--seed',type=int,default=4701)
    p.add_argument('--out',type=Path,default=Path('results/toy'))
    p.add_argument('--write-toy-trace',action='store_true',help='include toy labels and assumed q; never treat as real data')
    p2=sub.add_parser('replay',help='evaluate an explicit F0 JSONL with external error estimates')
    p2.add_argument('trace',type=Path)
    p2.add_argument('--out',type=Path,default=Path('results/replay'))
    p3=sub.add_parser('v07',help='offline adapter for documented V0.7 JSONL exports (schema integration unverified)')
    p3.add_argument('--inferences',type=Path,required=True)
    p3.add_argument('--scores',type=Path,required=True)
    p3.add_argument('--error-rates',type=Path,required=True,
                    help='JSON mapping agent ID to EXTERNAL estimated wrong probability')
    p3.add_argument('--out',type=Path,default=Path('results/v07'))
    p3.add_argument('--assumed-delay-ms',type=float,default=0.0)
    for cmd in (p,p2,p3):
        cmd.add_argument('--min-confidence',type=float,default=.90)
        cmd.add_argument('--max-source-error',type=float,default=.25)
        cmd.add_argument('--dependence-discount',type=float,default=1.0)
    args=parser.parse_args(argv)
    cfg=FusionConfig(min_score_confidence=args.min_confidence,
                     max_source_error=args.max_source_error,
                     dependence_discount=args.dependence_discount)
    if args.command=='toy':
        cases=generate(ToySettings(frames=args.frames,agents=args.agents,
                                   rois_per_frame=args.rois,seed=args.seed))
        estimator=QualityProxyEstimator()
        paired=tuple((c,estimator.estimate(c.observation)) for c in cases)
    elif args.command=='replay':
        paired=load_labeled_cases(args.trace)
    else:
        from .v07_adapter import load_v07_local_outputs
        supplied=json.loads(args.error_rates.read_text(encoding='utf-8'))
        if not isinstance(supplied,dict):
            parser.error('--error-rates must be a JSON object mapping agent IDs to error probabilities')
        paired=load_v07_local_outputs(args.inferences,args.scores,supplied,
                                      decision_delay_ms=args.assumed_delay_ms)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=compare(paired,cfg)
    save_comparison(args.out/'comparison.csv',rows)
    save_decisions(args.out/'selective_decisions.jsonl',paired,cfg)
    if args.command=='toy' and args.write_toy_trace:
        write_labeled_cases(args.out/'synthetic_labeled_trace.jsonl',paired)
    metadata={
        'kind':'synthetic' if args.command=='toy' else ('v07_offline_adapter_unverified' if args.command=='v07' else 'external_jsonl_not_independently_verified'),
        'baseline_comparison':'same decision-event observations for each policy',
        'confidence':'uncalibrated sigmoid of discounted evidence score',
        'evidence':'binary ROI labels; no raw sensor data, scores, logits or boxes',
        'energy_latency':'not simulated by F0; F1 integration required',
        'config':{k:getattr(cfg,k) for k in cfg.__dataclass_fields__},
        'events':len(paired),
    }
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))
    print('Saved',args.out)


if __name__=='__main__':
    main()
