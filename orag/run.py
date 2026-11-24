"""Run RAG / ORAG / no-retrieval entity typing with Llama-3-8B-Instruct.

usage: python -m orag.run --theme battery --corpus data/battery/corpus.jsonl \
           --out runs/battery_v1 --configs '[{"name": "a", "top_k": 10}, ...]'
Each config is run in <out>/<name>/ (records.jsonl, summary.json); the LLM and
the embedder are loaded once per job.
"""
import argparse
import re
import collections
import json
import os
import time

from .evaluate import aggregate, parse_types, score_entity
from .kb import chunk_text
from .llm import LLM, SYSTEMS, build_prompt, build_query
from .retrieval import ChunkRetriever, Embedder, UnitRetriever
from .themes import ROOT, THEME_TEXT, load_corpus, load_theme

DEFAULTS = {
    "embed_model": "Qwen/Qwen3-Embedding-8B",
    "llm_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "minilm_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_tokens": 600,
    "budget_tokens": 6000,
    "rag_max_chunks": None,
    "top_k": 10,
    "top_n_children": 3,
    "parent": True,
    "s_rule": "greater",
    "orag_max_items": None,
    "prompt_style": "knowledge",
    "system": "typing",
    "max_tokens": 160,
    "triple_tau": 0.2,
    "rel_tau": 0.35,
    "methods": ["rag", "orag"],
    "expand_eval": True,
    "retrieval_query": "full",
    "query_instruction": None,
    "children_in_k": "all",
    "examples": False,
    "natural_names": False,
    "wikidata_aliases": False,
    "constrained": False,
    "topk_types": False,
    "ontology_note": False,
    "child_depth": 1,
    "excluded_roots": None,
}


_ABBR = ("i.e", "e.g", "vs", "Fig", "Figs", "ca", "al", "Eq", "Ref", "approx", "resp", "No")


def _boundaries(passage, fix):
    """Sentence-end positions; with fix, a '.' between digits or after a common abbreviation is not an end."""
    import re
    out = []
    for m in re.finditer(r"[\n.。!?！？]", passage):
        j = m.start()
        if fix and passage[j] == ".":
            if j > 0 and j + 1 < len(passage) and passage[j - 1].isdigit() and passage[j + 1].isdigit():
                continue
            a, b = j, j + 1
            while a > 0 and (passage[a - 1].isalpha() or passage[a - 1] == "."):
                a -= 1
            while b < len(passage) and (passage[b].isalpha() or passage[b] == "."):
                b += 1
            tok = passage[a:b].strip(".")
            if tok in _ABBR or tok.lower() in ("i.e", "e.g", "et al", "cf"):
                continue
        out.append(j)
    return out


def sentence_of(passage, mention, fix=False):
    """The sentence (or line) of the passage containing the first occurrence of the mention."""
    import re
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(mention) + r"(?![A-Za-z0-9])", passage)
    i = m.start() if m else passage.find(mention)
    if i < 0:
        return passage
    b = _boundaries(passage, fix)
    start = max([j for j in b if j < i], default=-1) + 1
    end = min([j for j in b if j >= i + len(mention)], default=len(passage) - 1) + 1
    return passage[start:end].strip()


_BAT_HEAD = re.compile(r"(batter(y|ies)|cells?|RFBs?|LIBs?|SIBs?|PIBs?|KIBs?|ZIBs?|MIBs?|AIBs?|AORFBs?|NAORFBs?|"
                       r"VRBs?|VRFBs?|SRFBs?|chemistr(y|ies)|technolog(y|ies)|systems?|devices?)$", re.I)
_EL_HEAD = re.compile(r"(electrodes?|anodes?|cathodes?|foils?|felts?|plates?|wires?|mesh|cloth|papers?|"
                      r"materials?|metal|nanosheets?|films?|composites?|polymers?)$", re.I)


def route_mentions(jobs, llm, theme, fallback="llm"):
    """Sub-ontology (top-level type) of every mention: a head-word rule on the mention, then one constrained
    question to the LLM for undecided mentions. Battery only; other themes are not routed."""
    if theme != "battery":
        return [None] * len(jobs)
    routes, ask = [], []
    for i, (d, e, q) in enumerate(jobs):
        m = re.sub(r"\s*\([^)]*\)\s*$", "", e["mention"].strip())
        bat, el = bool(_BAT_HEAD.search(m)), bool(_EL_HEAD.search(m))
        routes.append("Battery" if bat and not el else "Electrode" if el and not bat else None)
        if routes[-1] is None:
            ask.append(i)
    if ask and fallback == "llm":
        prompts = [f"In the passage below, is the entity \"{jobs[i][1]['mention']}\" a battery (a battery, cell or "
                   f"battery technology) or an electrode (an electrode or electrode material)? Answer battery or "
                   f"electrode.\n{jobs[i][0]['passage']}" for i in ask]
        outs = llm.chat(prompts, system=None, max_tokens=4, allowed_labels=["battery", "electrode"], max_labels=1)
        for i, o in zip(ask, outs):
            routes[i] = "Electrode" if o.strip().lower().startswith("electrode") else "Battery"
    return routes


def retrieval_text(mode, d, e, query):
    """DPR query text: the full LLM query (default) or an entity-focused variant."""
    if mode == "full":
        return query
    name = e["mention"]
    fix = d.get("_query_fix", False)
    if mode == "entity_sentence":
        return f"{name}: {sentence_of(d['passage'], e['mention'], fix)}"
    if mode == "entity":
        return name
    if mode == "entity_query":
        return build_query(name, sentence_of(d["passage"], e["mention"], fix))
    raise ValueError(mode)


def summarize(records, methods):
    summary = collections.OrderedDict()
    for method in methods:
        rs = [r for r in records if r["method"] == method]
        summary[method] = {
            "all": aggregate([(r["P"], r["R"], r["F1"]) for r in rs]),
            "hard": aggregate([(r["P"], r["R"], r["F1"]) for r in rs if r["hard"]]),
            "all_flat": aggregate([tuple(r["flat"]) for r in rs]),
            "hard_flat": aggregate([tuple(r["flat"]) for r in rs if r["hard"]]),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rs) / max(1, len(rs)),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default='[{"name": "default"}]')
    args = ap.parse_args()
    configs = []
    for c in json.loads(args.configs):
        cfg = dict(DEFAULTS)
        cfg.update(c)
        if cfg["excluded_roots"] is None:
            cfg.pop("excluded_roots")
        configs.append(cfg)
    os.makedirs(args.out, exist_ok=True)

    llm = LLM(configs[0]["llm_model"])
    emb = Embedder(configs[0]["embed_model"], device=configs[0].get("embed_device", "cuda:1"),
                   query_instruction=configs[0].get("query_instruction"), max_len=configs[0].get("embed_max_len", 1024))
    minilm = Embedder(configs[0]["minilm_model"], device=configs[0].get("minilm_device", configs[0].get("embed_device", "cuda:1")),
                     batch_size=128, max_len=256)
    docs = load_corpus(args.corpus)
    cache = {}

    for cfg in configs:
        t0 = time.time()
        name = cfg.get("name", "default")
        theme_key = (cfg.get("triple_tau"), cfg.get("rel_tau"), tuple(cfg.get("excluded_roots", []) or []), cfg.get("natural_names", False), cfg.get("wikidata_aliases", False), cfg.get("game_kb_suffix", ""), cfg.get("game_rag_plain", False), cfg.get("battery_rag", "enriched"), cfg.get("paper_aliases", False), cfg.get("parent_line", True), cfg.get("placeholder", True), cfg.get("rag_shared_lines", False), cfg.get("kb_enrichment", False))
        if ("theme", theme_key) not in cache:
            onto, units, rag_lines = load_theme(args.theme, cfg, minilm)
            parents = {t: set(onto.parents.get(t, ())) for t in onto.types}
            children = {t: list(onto.children.get(t, [])) for t in onto.types}
            for n, u in units.items():
                for p in u.get("parents", []):
                    parents.setdefault(n, set()).add(p)
                    children.setdefault(p, [])
                    if n not in children[p]:
                        children[p].append(n)
            orag = UnitRetriever(units, parents, children, emb, llm.count_tokens)
            cache[("theme", theme_key)] = (onto, units, rag_lines, orag)
        onto, units, rag_lines, orag = cache[("theme", theme_key)]
        ck = ("rag", cfg["chunk_tokens"], theme_key)
        if ck not in cache:
            chunks = chunk_text(rag_lines, llm.count_tokens, cfg["chunk_tokens"])
            cache[ck] = ChunkRetriever(chunks, emb, llm.count_tokens)
        rag = cache[ck]

        for d in docs:
            d["_query_fix"] = cfg.get("query_fix", False)
        jobs = [(d, e, build_query(e["mention"], d["passage"]))
                for d in docs for e in d["entities"]]
        rq = cfg.get("retrieval_query", "full")
        qk = ("q", args.corpus, rq, cfg.get("query_fix", False))
        if qk not in cache:
            if rq == "mix":  # average of the entity-in-sentence query and the bare mention
                import numpy as np
                a = emb.encode_queries([retrieval_text("entity_query", d, e, q) for d, e, q in jobs])
                b = emb.encode_queries([retrieval_text("entity", d, e, q) for d, e, q in jobs])
                v = a + b
                cache[qk] = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")
            else:
                cache[qk] = emb.encode_queries([retrieval_text(rq, d, e, q) for d, e, q in jobs])
        qvecs = cache[qk]
        system = None
        if cfg["system"]:
            tpl = SYSTEMS[cfg["system"]]
            if not cfg.get("system_theme", True):  # the paper's prompt names no theme
                tpl = tpl.replace(' on the theme "{theme}"', "")
            system = tpl.format(theme=THEME_TEXT[args.theme])
        if system and cfg.get("ontology_note"):
            from .llm import ONTOLOGY_NOTES
            system = system + " " + ONTOLOGY_NOTES[args.theme]
        if system and cfg.get("examples"):
            from .llm import EXAMPLES
            system = system + " " + EXAMPLES[args.theme + ("_v2" if cfg.get("examples") == "v2" else "")]

        records = []
        routes = None
        if cfg.get("grammar_tree") or cfg.get("orag_route"):
            rk = ("route", args.corpus, args.theme, cfg.get("route_fallback", "llm"))
            if rk not in cache:
                cache[rk] = route_mentions(jobs, llm, args.theme, cfg.get("route_fallback", "llm"))
            routes = cache[rk]
        tree_nodes = {}
        if routes is not None:
            for root in onto.roots:
                tree_nodes[root] = {t for t in onto.types if t == root or root in onto.ancestors(t)}

        for method in cfg["methods"]:
            prompts, metas = [], []
            for (d, e, q), qv in zip(jobs, qvecs):
                if method == "rag":
                    ev = rag.retrieve(qv, cfg.get("rag_budget_tokens") or cfg["budget_tokens"], cfg["rag_max_chunks"])
                    meta = {"chunk_ids": [x["chunk_id"] for x in ev]}
                    style = cfg["prompt_style"]
                elif method == "orag":
                    ev, start = orag.retrieve(qv, cfg["budget_tokens"], cfg["top_k"], cfg["top_n_children"],
                                              cfg["parent"], cfg["s_rule"], cfg["orag_max_items"],
                                              cfg.get("children_in_k", "all"), cfg.get("topk_types", False),
                                              cfg.get("child_depth", 1), cfg.get("channels", "joint"),
                                              cfg.get("top_k_s"),
                                              tree_nodes.get(routes[len(prompts)]) if (cfg.get("orag_route") and routes) else None)
                    meta = {"start": start, "items": [(x["node"], x["kind"], x["role"]) for x in ev]}
                    style = cfg["prompt_style"]
                elif method == "oracle":  # diagnostic: units of the gold types (+ parents, siblings)
                    nodes = []
                    for g in e["gold"]:
                        for n in [g] + sorted(parents.get(g, ())):
                            if n not in nodes:
                                nodes.append(n)
                        for p_ in parents.get(g, ()):
                            for c in children.get(p_, [])[:6]:
                                if c not in nodes:
                                    nodes.append(c)
                    ev, used = [], 0
                    for n in nodes:
                        t = units[n]["K"] if n in units else ""
                        if t and used + llm.count_tokens(t) <= cfg["budget_tokens"]:
                            ev.append({"text": t}); used += llm.count_tokens(t)
                    meta, style = {"nodes": nodes}, cfg["prompt_style"]
                elif method == "labels":  # context + the bare label hierarchy (no theme knowledge)
                    ev, meta, style = [{"text": "\n".join(onto.tree_lines)}], {}, cfg["prompt_style"]
                else:  # context only
                    ev, meta, style = [], {}, "none"
                prompts.append(build_prompt(style, q, ev))
                metas.append(meta)
            allowed = [t for t in onto.types if t not in onto.excluded] if cfg.get("constrained") else None
            if allowed is not None and cfg.get("grammar_roots"):
                allowed = list(onto.types)  # roots may be generated; scoring leaves them out
            if allowed is not None and cfg.get("grammar_tree") and routes is not None:
                allowed = [[t for t in allowed if routes[i] is None or t in tree_nodes[routes[i]]]
                           for i in range(len(prompts))]
            if cfg.get("constrained") == "evidence" and not (allowed and isinstance(allowed[0], list)):
                # decode only type names that occur in this prompt's retrieved knowledge (the system prompt asks
                # for names "exactly as they appear in the knowledge"); prompts without any fall back to all
                import re as _re
                names = set(allowed)
                per = []
                for p_ in prompts:
                    ev_txt = p_.split("Knowledge:", 1)[-1]
                    found = sorted({w for w in _re.findall(r"[A-Za-z][A-Za-z0-9]*", ev_txt) if w in names})
                    per.append(found or allowed)
                allowed = per
            outs = llm.chat(prompts, system=system, max_tokens=cfg["max_tokens"], allowed_labels=allowed,
                            temperature=cfg.get("temperature", 0.0), top_p=cfg.get("top_p", 1.0),
                            seed=cfg.get("seed", 0))
            for (d, e, q), p, o, m in zip(jobs, prompts, outs, metas):
                pred = parse_types(o, onto)
                s = score_entity(pred, e["gold"], onto, expand=cfg["expand_eval"])
                s_flat = score_entity(pred, e["gold"], onto, expand=False)
                records.append({"method": method, "doc_id": d["doc_id"], "mention": e["mention"],
                                "hard": e.get("hard", False), "gold": e["gold"], "output": o, "pred": pred,
                                "P": s[0], "R": s[1], "F1": s[2], "flat": s_flat,
                                "prompt_tokens": llm.count_tokens(p), "prompt": p, "meta": m})
        summary = summarize(records, cfg["methods"])
        od = os.path.join(args.out, name)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "records.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(od, "summary.json"), "w") as f:
            json.dump({"theme": args.theme, "corpus": args.corpus, "config": cfg, "system_prompt": system, "summary": summary, "n_chunks": len(rag.chunks),
                       "n_units": len(units), "seconds": time.time() - t0}, f, indent=1)
        line = {m: {k: round(summary[m][k].get(x, 0), 3) for k in ("all", "hard") for x in ("P",)}
                for m in cfg["methods"]}
        print("CONFIG", name, json.dumps({m: {k: [round(summary[m][k].get(x, 0), 3) for x in ("P", "R", "F1")]
                                             for k in ("all", "hard")} for m in cfg["methods"]}), flush=True)


if __name__ == "__main__":
    main()
