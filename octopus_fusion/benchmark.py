"""Fair paired synthetic/JSONL baselines with identical input cases.

Do not interpret any gains from this toy as V2V4Real or real-system results.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from .contracts import Action, FusionConfig, FusionDecision, ReliabilityEstimate
from .evaluate import LabeledCase, evaluate
from .fusion import SelectiveFuser, unweighted_majority, weighted_vote


def _baseline(case: LabeledCase, label: int | None, method: str) -> FusionDecision:
    obs=case.observation
    agents=tuple((x.agent_id,x.prediction) for x in sorted(obs.evidence,key=lambda x:x.agent_id))
    if label is None:
        return FusionDecision(obs.event_id,obs.frame_id,Action.KEEP_LOCAL,
                              agents,None,0,.5,(),(),(),('tie',),method)
    return FusionDecision(obs.event_id,obs.frame_id,Action.SHARE_FUSED,
                          tuple((agent,label) for agent,_ in agents),label,0,.5,
                          (),(),(),(),method)


def compare(paired: tuple[tuple[LabeledCase,ReliabilityEstimate],...],
            config: FusionConfig | None = None) -> tuple[dict,...]:
    if not paired:
        raise ValueError('Require cases')
    cases=tuple(pair[0] for pair in paired)
    fuser=SelectiveFuser(config)
    modes={m:{} for m in ('local_only','majority','weighted_naive','selective_f0')}
    for case,reliability in paired:
        obs=case.observation
        modes['local_only'][obs.event_id]=_baseline(case,None,'local_only')
        modes['majority'][obs.event_id]=_baseline(case,unweighted_majority(obs),'majority')
        modes['weighted_naive'][obs.event_id]=_baseline(case,weighted_vote(obs,reliability),'weighted_naive')
        modes['selective_f0'][obs.event_id]=fuser.decide(obs,reliability)
    rows=[]
    for name,actions in modes.items():
        m=evaluate(cases, actions)
        rows.append({
            'policy':name,'events':m.cases,
            'marginal_error_rate':m.average_agent_error,
            'joint_all_wrong_rate':m.all_agent_error_rate,
            'shared_coverage':m.shared_coverage,
            'accepted_fusion_error_rate':m.accepted_fusion_error_rate,
            'fallback_rate':m.fallback_rate,'mask_and_fuse_rate':m.mask_and_fuse_rate,
        })
    return tuple(rows)


def save_comparison(path: Path, rows: tuple[dict,...]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_decisions(path: Path,
                   paired: tuple[tuple[LabeledCase,ReliabilityEstimate],...],
                   config: FusionConfig | None=None) -> None:
    """Decisions-only export: NO ground truth or correctness in this file."""
    path.parent.mkdir(parents=True,exist_ok=True)
    fuser=SelectiveFuser(config)
    with path.open('w',encoding='utf-8') as f:
        for case,rel in paired:
            d=fuser.decide(case.observation,rel)
            from dataclasses import asdict
            f.write(json.dumps(asdict(d),sort_keys=True)+'\n')
