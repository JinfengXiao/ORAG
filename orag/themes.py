"""Theme definitions: ontology, structured units (ORAG) and flat memory (RAG)."""
import json
import os
import re

from .ontology import Ontology
from .kb import parse_battinfo_enriched, battery_units, load_units

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, "data", *p)

THEME_TEXT = {
    "battery": "redox-active organic electrode materials",
    "game": "online game posts for Genshin Impact",
}


def load_battery(cfg, minilm=None):
    onto = Ontology([D("battery", "ontology_Battery.txt"), D("battery", "ontology_Electrode.txt")],
                    theme_roots_excluded=cfg.get("excluded_roots") or ["Battery", "Electrode"])
    kf = D("battery", "battinfo-enriched.txt")
    if not os.path.exists(kf):
        raise FileNotFoundError("data/battery/battinfo-enriched.txt is missing: build the battery knowledge file "
                                "as described in README.md, section 2.")
    info, raw_lines = parse_battinfo_enriched(kf)
    # aliases help map LLM outputs written as alias names back to types
    for t, d in info.items():
        if t in onto.type_set:
            for a in d["alias"]:
                for x in a.split(","):
                    onto.add_alias(x.strip(), t)
    keep = None
    tau = cfg.get("triple_tau")
    if tau is not None and minilm is not None:
        trip = [(t, s, p, o) for t, d in info.items() if t in onto.type_set for (s, p, o) in d["triples"]]
        from .kb import triple_sentence
        sents = [triple_sentence(s, p, o) for (_, s, p, o) in trip]
        import numpy as np
        ev = minilm.encode(sents)
        th = minilm.encode([THEME_TEXT["battery"]])[0]
        sims = ev @ th
        # relation filter (an addition of this implementation): similarity of the relation to pre-defined relations
        rel_tau = cfg.get("rel_tau")
        rels = sorted({p for (_, _, p, _) in trip})
        rv = minilm.encode(rels)
        pv = minilm.encode(["is a", "has property of", "has instances"])
        rsim = dict(zip(rels, (rv @ pv.T).max(axis=1)))
        kept = {(s, p, o) for (_, s, p, o), v in zip(trip, sims)
                if v >= tau and (rel_tau is None or rsim[p] >= rel_tau)}
        keep = lambda s, p, o: (s, p, o) in kept
    # optional Wikidata enrichment (labels/aliases of the Wikidata items linked from the types),
    # added to the shared knowledge of both ORAG and RAG
    wd_lines = {}
    if cfg.get("wikidata_aliases"):
        wd = json.load(open(D("battery", "wikidata_type_aliases.json"), encoding="utf-8"))
        for t, w in wd.items():
            names = [n for n in [w.get("label", "")] + w.get("aliases", []) if n and len(n) < 60]
            if t in onto.type_set and names:
                info.setdefault(t, {"definition": [], "alias": [], "children": [], "descriptions": [], "triples": []})
                info[t]["alias"].append(", ".join(dict.fromkeys(names)))
                wd_lines[t] = f"{t} is also known as {', '.join(dict.fromkeys(names))}"
                for n in names:
                    onto.add_alias(n, t)
    if cfg.get("paper_aliases"):
        from .kb import PAPER_ALIASES
        for t, names in PAPER_ALIASES.items():
            info.setdefault(t, {"definition": [], "alias": [], "children": [], "descriptions": [], "triples": []})
            info[t]["alias"].append(", ".join(names))
            wd_lines[t] = (wd_lines[t] + ", " if t in wd_lines else f"{t} is also known as ") + ", ".join(names)
            for n in names:
                onto.add_alias(n, t)
    extra = None
    if cfg.get("kb_enrichment"):
        extra = json.load(open(D("battery", "kb_enrichment.json"), encoding="utf-8"))["types"]
    units = battery_units(onto, info, keep_triple=keep, natural_names=cfg.get("natural_names", False),
                          parent_line=cfg.get("parent_line", True), placeholder=cfg.get("placeholder", True),
                          extra=extra)
    # flat memory for vanilla RAG = the enriched knowledge file as text (+ the same Wikidata lines)
    rag_lines = []
    shared = cfg.get("rag_shared_lines", False)
    from .ontology import camel_to_words
    for line in raw_lines:
        m0 = re.match(r"^(\w+) means: (.*)$", line)
        if shared and m0 and m0.group(1) in onto.type_set:
            # the same sentences ORAG's knowledge units add (name gloss, parent types)
            t0 = m0.group(1)
            if cfg.get("natural_names", False):
                line = f"{t0} ({camel_to_words(t0)}) means: {m0.group(2)}"
            rag_lines.append(line)
            pars = sorted(onto.parents.get(t0, ()))
            if pars and cfg.get("parent_line", True):
                rag_lines.append(f"{t0} is a type of {', '.join(pars)}")
            if t0 in wd_lines:
                rag_lines.append(wd_lines.pop(t0))
            continue
        rag_lines.append(line)
        m = re.match(r"^(\w+) means: ", line)
        if m and m.group(1) in wd_lines:
            rag_lines.append(wd_lines.pop(m.group(1)))
    rag_lines += list(wd_lines.values())
    if extra:  # the same added knowledge goes into the RAG memory, after each type's definition
        out, done = [], set()
        for line in rag_lines:
            out.append(line)
            m1 = re.match(r"^(\w+)( \([^)]*\))? means: ", line)
            if m1 and m1.group(1) in extra and m1.group(1) not in done:
                done.add(m1.group(1))
                out += extra[m1.group(1)].get("K", []) + extra[m1.group(1)].get("S", [])
        out += [x for t, v in extra.items() if t not in done for x in v.get("K", []) + v.get("S", [])]
        rag_lines = out
    mode = cfg.get("battery_rag", "enriched")
    if mode != "enriched":
        # same knowledge, without the theme-hierarchy curation: type blocks in the order of the raw
        # BattINFO serialization (battinfo-inferred.ttl, IRI order) and optionally without the
        # derived "includes following types" lines
        order = ttl_order()
        blocks, cur, key = {}, [], None
        for line in rag_lines:
            m = re.match(r"^(\w+)( means: | is also known as | includes following types: |:\s|, )", line)
            k = m.group(1) if m else key
            if k != key and cur:
                blocks.setdefault(key, []).extend(cur); cur = []
            key = k
            if mode == "ttl_nochildren" and " includes following types: " in line:
                continue
            cur.append(line)
        if cur:
            blocks.setdefault(key, []).extend(cur)
        keys = sorted(blocks, key=lambda t: order.get(t, 10 ** 9))
        rag_lines = [l for t in keys for l in blocks[t]]
    return onto, units, rag_lines


def ttl_order():
    """Position of each class label in the raw BattINFO serialization."""
    path = os.environ.get("ORAG_BATTINFO_TTL") or D("battery", "battinfo-inferred.ttl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: download battinfo-inferred.ttl from BattINFO "
                                "(README.md, section 2) or set ORAG_BATTINFO_TTL to its path.")
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8")):
        m = re.search(r'skos/core#prefLabel> "([^"]+)"', line)
        if m and m.group(1) not in out:
            out[m.group(1)] = i
    return out


def load_game(cfg, minilm=None):
    onto = Ontology([D("game", "ontology.txt")], theme_roots_excluded=cfg.get("excluded_roots") or [])
    with open(D("game", "type_aliases.json"), encoding="utf-8") as f:
        for t, al in json.load(f).items():
            for a in al:
                onto.add_alias(a, t)
    sfx = cfg.get("game_kb_suffix", "")
    units = load_units(D("game", f"units{sfx}.jsonl"))
    import sys
    for n, u in units.items():   # checks of a user-built knowledge base
        bad = [p for p in u.get("parents", []) if p not in onto.type_set]
        if bad:
            print(f"warning: unit {n!r}: parents {bad} are not ontology types", file=sys.stderr)
        if u.get("parents") and n in onto.type_set:
            print(f"warning: entity unit {n!r} has the name of an ontology type and replaces its type unit", file=sys.stderr)
    rag_file = {True: f"rag_memory_plain{sfx}.txt", "plain": f"rag_memory_plain{sfx}.txt",
                "wiki": f"rag_memory_wiki{sfx}.txt"}.get(cfg.get("game_rag_plain"), f"rag_memory{sfx}.txt")
    with open(D("game", rag_file), encoding="utf-8") as f:
        rag_lines = [l.rstrip("\n") for l in f]
    return onto, units, rag_lines


def load_theme(name, cfg, minilm=None):
    return {"battery": load_battery, "game": load_game}[name](cfg, minilm)


def load_corpus(path):
    import sys
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    for d in docs:   # the code finds each mention's sentence by its first occurrence not next to an ASCII letter or digit
        for e in d.get("entities", []):
            if e["mention"] not in d.get("passage", ""):
                print(f"warning: {d['doc_id']}: mention {e['mention']!r} is not in the passage", file=sys.stderr)
    return docs
