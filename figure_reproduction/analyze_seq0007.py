"""Class-conditioned, non-oracle policies on the recorded 0007 trace."""
from pathlib import Path
import json,csv
from collections import Counter
root=Path(__file__).resolve().parent
r=[json.loads(x) for x in (root/'sources/v2v4real_seq0007_trace.jsonl').read_text().splitlines()]
assert len(r)==442 and len({x['frame_id'] for x in r})==221
counts=Counter(); outcomes=[]; disagree_frames=set()
for v in r:
    y=v['truth']; p=v['candidates']; a,b,c=[p[k] for k in ('local:ego','local:agent1','edge:ego')]
    assert c==p['edge:agent1'] and all(x in (0,1) for x in [a,b,c,y])
    if a!=b: disagree_frames.add(v['frame_id']);counts[(y,'disagree')]+=1;counts[(y,'candidate_correct_on_disagree')]+=int(c==y)
    g=c if a!=b else a
    assert g==max(a,b), 'Disagreement gate differs from OR on this trace'
    for name,labels in {'local':(a,b),'shared':(c,c),'disagreement_gate':(g,g),'OR':(max(a,b),)*2,'AND':(min(a,b),)*2}.items():
        errors=sum(int(x!=y) for x in labels)
        outcomes.append((name,y,errors))
with (root/'sources/seq0007_class_policy.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['policy','truth','queries','agent_errors','joint_errors','marginal_error','joint_error'])
    for name in ['local','shared','disagreement_gate','OR','AND']:
        for y in [1,0]:
            e=[k for n,t,k in outcomes if n==name and t==y];w.writerow([name,y,len(e),sum(e),e.count(2),sum(e)/(2*len(e)),e.count(2)/len(e)])
print('Class-disagreement counts:',dict(counts),'frames:',len(disagree_frames))
print((root/'sources/seq0007_class_policy.csv').read_text())
