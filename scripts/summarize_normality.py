"""Reduce replication-level output to Gaussian diagnostics (standard library only)."""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

def quantile(x, p):
    y = sorted(x); h = (len(y)-1)*p; lo = int(h); hi = min(lo+1, len(y)-1)
    return y[lo] + (h-lo)*(y[hi]-y[lo])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    groups=defaultdict(list); files=sorted(Path(a.input).rglob('*.csv'))
    for path in files:
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                if r.get('status')!='success' or r.get('method')!='naive' or not r.get('estimate'): continue
                key=(r['scenario'],int(r['n']),float(r['coordinate']),r['coordinate_name'])
                groups[key].append(math.sqrt(int(r['n']))*(float(r['estimate'])-float(r['target'])))
    probs=[.01,.025,.05,.10,.25,.50,.75,.90,.95,.975,.99]; rows=[]; nd=NormalDist()
    for key,x in groups.items():
        n=len(x)
        if n<20: continue
        mean=sum(x)/n; sd=math.sqrt(sum((v-mean)**2 for v in x)/(n-1)); z=sorted((v-mean)/sd for v in x)
        skew=n/((n-1)*(n-2))*sum(v**3 for v in z)
        kurt=(n*(n+1)/((n-1)*(n-2)*(n-3))*sum(v**4 for v in z)-3*(n-1)**2/((n-2)*(n-3))) if n>3 else float('nan')
        ks=max(max((i+1)/n-nd.cdf(v), nd.cdf(v)-i/n) for i,v in enumerate(z))
        row={'scenario':key[0],'n':key[1],'coordinate':key[2],'coordinate_name':key[3],'replications':n,'mean_root_n_error':mean,'sd_root_n_error':sd,'skewness':skew,'excess_kurtosis':kurt,'ks_distance':ks}
        row.update({f'q{int(1000*p):03d}':quantile(z,p) for p in probs}); rows.append(row)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} coordinate diagnostics from {len(files)} files to {out}')
if __name__=='__main__': main()
