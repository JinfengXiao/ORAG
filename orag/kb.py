"""Structured retrieval memory.

Every ontology node gets a structured knowledge unit with two sub-units:
  * knowledge-dense (K): definition, alias, KG triples, instances, sub-types
  * semantic-rich (S): free-text descriptions (Wikipedia / game wiki)
All information is restated as complete sentences (paper, Sec. 3.2.2).

The vanilla-RAG memory is the same enriched knowledge as flat text, split into
fixed-size chunks (~600 tokens).
"""
import collections
import json
import re

from .ontology import camel_to_words

# ---------------------------------------------------------------- battery KB

_REL_TEMPLATES = {
    "subclass of": "{s} is a subclass of {o}.",
    "instance of": "{s} is an instance of {o}.",
    "part of": "{s} is part of {o}.",
    "has part(s)": "{s} has part {o}.",
    "has part": "{s} has part {o}.",
    "has characteristic": "{s} has characteristic {o}.",
    "has use": "{s} has use {o}.",
    "uses": "{s} uses {o}.",
    "source of energy": "{s} has source of energy {o}.",
    "opposite of": "{s} is the opposite of {o}.",
    "different from": "{s} is different from {o}.",
    "said to be the same as": "{s} is said to be the same as {o}.",
    "facet of": "{s} is a facet of {o}.",
    "has effect": "{s} has effect {o}.",
    "discoverer or inventor": "{s} was invented by {o}.",
    "named after": "{s} is named after {o}.",
    "time of discovery or invention": "{s} was invented in {o}.",
    "made from material": "{s} is made from {o}.",
    "product or material produced or service provided": "{s} produces {o}.",
}

# Obvious copy-paste error in BattINFO (definition duplicated from the metallic
# sibling); corrected here.
BATTINFO_FIXES = {
    "NonAqueousOrganicFlowBattery means: a full redox flow battery with non-aqueous electrolytes that contain metallic species as the active components":
    "NonAqueousOrganicFlowBattery means: a full redox flow battery with non-aqueous electrolytes that contain organic species as the active components",
}


def triple_sentence(s, p, o):
    tpl = _REL_TEMPLATES.get(p.strip().lower())
    if tpl:
        return tpl.format(s=s, o=o)
    return f"{s} {p.strip()} {o}."


def parse_battinfo_enriched(path):
    """Return {type: {'definition','alias','children','descriptions','triples'}}."""
    info = collections.defaultdict(lambda: {"definition": [], "alias": [], "children": [],
                                            "descriptions": [], "triples": []})
    raw_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            line = BATTINFO_FIXES.get(line, line)
            raw_lines.append(line)
            if not line.strip():
                continue
            m = re.match(r"^(\w+) means: (.*)$", line)
            if m:
                info[m.group(1)]["definition"].append(m.group(2).strip())
                continue
            m = re.match(r"^(\w+) is also known as (.*)$", line)
            if m:
                info[m.group(1)]["alias"].append(m.group(2).strip())
                continue
            m = re.match(r"^(\w+) includes following types: (.*)$", line)
            if m:
                kids = [k.strip() for k in m.group(2).split(",") if k.strip()]
                info[m.group(1)]["children"].extend(kids)
                continue
            m = re.match(r"^(\w+):\s+(.*)$", line)
            if m:
                info[m.group(1)]["descriptions"].append(m.group(2).strip())
                continue
            m = re.match(r"^(\w+), ([^,]+), (.*)$", line)
            if m:
                info[m.group(1)]["triples"].append((m.group(1), m.group(2).strip(), m.group(3).strip()))
                continue
            raise ValueError("unparsed line: " + line[:200])
    return dict(info), raw_lines


# Alias given verbatim as the paper's example of a restated fact (Sec. 3.2.2: "Redox flow battery
# is also known as RFB"); BattINFO gives the type no alternative label.
PAPER_ALIASES = {"RedoxFlowBattery": ["RFB"]}


def battery_units(onto, info, keep_triple=None, natural_names=False, parent_line=True, placeholder=True, extra=None):
    """Build structured units for all ontology types.

    keep_triple: optional predicate (s, p, o) -> bool implementing the
    theme-similarity filter on KG triples.
    parent_line: add "X is a type of <parents>" (not in the knowledge file).
    placeholder: with natural_names, give types without a definition the sentence
    "X (words) is a type in the theme ontology." (not in the knowledge file).
    extra: optional {type: {"K": [sentences], "S": [paragraphs]}} appended to the units.
    """
    units = {}
    for t in onto.types:
        d = info.get(t, {"definition": [], "alias": [], "children": [], "descriptions": [], "triples": []})
        k = []
        words = camel_to_words(t)
        for x in d["definition"]:
            k.append(f"{t} ({words}) means: {x}" if natural_names else f"{t} means: {x}")
        if natural_names and placeholder and not d["definition"]:
            k.append(f"{t} ({words}) is a type in the theme ontology.")
        for x in d["alias"]:
            x = re.sub(r'"+@en\s*"?|"', "", x).strip()
            if x:
                k.append(f"{t} is also known as {x}")
        pars = sorted(onto.parents.get(t, ()))
        if pars and parent_line:
            k.append(f"{t} is a type of {', '.join(pars)}")
        kids = onto.children.get(t) or d["children"]
        if kids:
            k.append(f"{t} includes following types: {', '.join(kids)}")
        for (s, p, o) in d["triples"]:
            if keep_triple is None or keep_triple(s, p, o):
                k.append(triple_sentence(s, p, o))
        s_text = " ".join(f"{t}: {x}" for x in d["descriptions"])
        if extra and t in extra:
            k += extra[t].get("K", [])
            s_text = " ".join([s_text] + extra[t].get("S", [])).strip()
        units[t] = {"type": t, "K": "\n".join(k), "S": s_text}
    return units


# ------------------------------------------------------------------ chunking

def chunk_text(lines, tokenizer_len, chunk_tokens=600):
    """Greedy fixed-size chunking over lines (lines longer than the budget are
    split on whitespace)."""
    chunks, cur, cur_len = [], [], 0
    pieces = []
    for line in lines:
        if not line.strip():
            continue
        n = tokenizer_len(line)
        if n <= chunk_tokens:
            pieces.append((line, n))
        else:
            words = line.split(" ")
            buf = []
            for w in words:
                buf.append(w)
                if tokenizer_len(" ".join(buf)) >= chunk_tokens:
                    pieces.append((" ".join(buf), tokenizer_len(" ".join(buf))))
                    buf = []
            if buf:
                pieces.append((" ".join(buf), tokenizer_len(" ".join(buf))))
    for text, n in pieces:
        if cur and cur_len + n > chunk_tokens:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(text)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def save_units(units, path):
    with open(path, "w", encoding="utf-8") as f:
        for u in units.values():
            f.write(json.dumps(u, ensure_ascii=False) + "\n")


def load_units(path):
    import sys
    units = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u = json.loads(line)
            if u["type"] in units:
                print(f"warning: unit {u['type']!r} appears more than once; the last one is kept", file=sys.stderr)
            units[u["type"]] = u
    return units
