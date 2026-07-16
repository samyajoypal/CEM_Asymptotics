"""Audit cited BibTeX records against Crossref and emit a machine-readable report."""
from __future__ import annotations
import json, re, time, unicodedata, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def entries(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)(?=\n@|\Z)", text, re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?", m.group(3), re.S):
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", f.group(2)).strip()
        out[m.group(2)] = fields
    return out

def cited_keys(aux):
    keys=[]
    for group in re.findall(r"\\citation\{([^}]*)\}", aux):
        for key in group.split(","):
            if key not in keys: keys.append(key)
    return keys

def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":"CEM-reference-audit/1.0 (mailto:samyajoy.pal@rptu.de)"})
    with urllib.request.urlopen(req,timeout=30) as r: return json.load(r)["message"]

def main():
    bib=entries((ROOT/"paper/references.bib").read_text())
    keys=cited_keys((ROOT/"paper/jrssb_main.aux").read_text())
    rows=[]
    for key in keys:
        b=bib[key]; doi=b.get("doi")
        try:
            if doi: rec=get("https://api.crossref.org/works/"+urllib.parse.quote(doi,safe=""))
            else:
                q=urllib.parse.quote(b.get("title","")+" "+b.get("author","")+" "+b.get("year",""))
                rec=get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=1")["items"][0]
            ct=(rec.get("title") or [""])[0]; cy=str((rec.get("published-print") or rec.get("published") or rec.get("issued"))["date-parts"][0][0])
            ca=" and ".join(" ".join([a.get("given",""),a.get("family","")]) for a in rec.get("author",[]))
            title_ok=norm(b.get("title",""))==norm(ct); year_ok=b.get("year")==cy
            score=rec.get("score",1.0) if not doi else 1.0
            rows.append(dict(key=key,status="verified" if title_ok and year_ok and score>.7 else "review",
                doi_bib=doi or "",doi_crossref=rec.get("DOI",""),title_bib=b.get("title",""),title_crossref=ct,
                year_bib=b.get("year",""),year_crossref=cy,authors_crossref=ca,
                container_crossref=(rec.get("container-title") or [""])[0],volume_crossref=rec.get("volume",""),
                issue_crossref=rec.get("issue",""),pages_crossref=rec.get("page",""),score=score))
        except Exception as exc:
            rows.append(dict(key=key,status="error",error=str(exc)))
        time.sleep(.08)
    out=ROOT/"docs/reference_audit.json"; out.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    print(f"wrote {len(rows)} records; status counts:", {s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)})

if __name__=="__main__": main()
