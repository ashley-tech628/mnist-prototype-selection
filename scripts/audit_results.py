"""Recompute saved summaries and paired differences without rerunning MNIST."""
from pathlib import Path
import csv
import json
import math
import statistics
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def audit():
    rows=list(csv.DictReader((ROOT/'results/historical/results.csv').open(newline='')))
    summary=list(csv.DictReader((ROOT/'results/historical/summary.csv').open(newline='')))
    expected={(s['method'],int(s['M'])):s for s in summary}
    groups={}
    for r in rows:
        key=(r['method'],int(r['M']))
        group=groups.setdefault(key,{})
        t=int(r['trial'])
        if t in group:raise ValueError('Duplicate trial')
        value=float(r['acc'])
        assert 0<=value<=1
        group[t]=(int(r['seed']),value)
    for key,group in groups.items():
        values=[v[1] for v in group.values()]
        s=expected[key]
        assert len(values)==10==int(s['n_trials'])
        mean=statistics.mean(values);std=statistics.stdev(values)
        assert math.isclose(mean,float(s['mean_acc']),abs_tol=1e-12)
        assert math.isclose(std,float(s['std_acc']),abs_tol=1e-12)
        assert math.isclose(2.262*std/math.sqrt(10),float(s['ci95_halfwidth']),abs_tol=1e-12)
    paired=[]
    for budget in sorted({key[1] for key in groups}):
        random=groups[('random',budget)];cluster=groups[('kmeans_per_class',budget)]
        assert random.keys()==cluster.keys()
        assert all(random[t][0]==cluster[t][0] for t in random)
        delta=[cluster[t][1]-random[t][1] for t in sorted(random)]
        mean=statistics.mean(delta);half=2.262*statistics.stdev(delta)/math.sqrt(len(delta))
        paired.append({'M':budget,'n_pairs':len(delta),'gain_percentage_points':100*mean,
                       'paired_t95_interval_percentage_points':[100*(mean-half),100*(mean+half)]})
    for record in json.loads((ROOT/'docs/source-manifest.json').read_text()):
        if 'packaged_path' in record:
            assert hashlib.sha256((ROOT/record['packaged_path']).read_bytes()).hexdigest()==record['sha256']
    return {'recorded_trials':len(rows),'paired_results':paired,
            'interpretation':'Intervals describe prototype-selection seed variation on a fixed test set, not population uncertainty.'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    result=audit()
    if a.write:(ROOT/'results/historical/paired_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
