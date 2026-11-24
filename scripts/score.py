"""Score RAG and ORAG answers.

usage: python scripts/score.py --theme battery --corpus data/battery/corpus.jsonl \
           --records runs/battery/battery_rag/records.jsonl runs/battery/battery_orag/records.jsonl
Reads the records written by `python -m orag.run` and prints, as JSON, the macro P/R/F1 of RAG and ORAG on all
entities of the corpus and on its hard cases. For the battery theme pass the records of both configurations:
battery_rag supplies the RAG answers and battery_orag the ORAG answers.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from orag.evaluate import parse_types, score_entity  # noqa: E402
from orag.scoring import SCORING, load_onto  # noqa: E402
from orag.themes import load_corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--theme", choices=("battery", "game"), required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    a = ap.parse_args()
    onto = load_onto(a.theme, *SCORING[a.theme])
    answers = {}
    for p in a.records:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["method"] in ("rag", "orag"):
                answers.setdefault((r["method"], r["doc_id"], r["mention"]), r["output"])
    rows = {}
    missing = 0
    for d in load_corpus(a.corpus):
        for e in d["entities"]:
            for m in ("rag", "orag"):
                out = answers.get((m, d["doc_id"], e["mention"]))
                if out is None:
                    missing += 1
                    continue
                s = score_entity(parse_types(out, onto), e["gold"], onto)
                rows.setdefault(f"{m}/all", []).append(s)
                if e.get("hard"):
                    rows.setdefault(f"{m}/hard", []).append(s)
    res = {k: {"n": len(v), "P": round(sum(x[0] for x in v) / len(v), 4), "R": round(sum(x[1] for x in v) / len(v), 4),
               "F": round(sum(x[2] for x in v) / len(v), 4)} for k, v in sorted(rows.items())}
    if missing:
        res["missing_answers"] = missing
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
