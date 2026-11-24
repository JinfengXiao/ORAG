"""Output parsing and per-entity hierarchical precision / recall / F1.

Protocol (ancestor rule: paper Sec. 3.1; macro-average: Sec. 4.3): per entity, P = |pred ∩
gold| / |pred|, R = |pred ∩ gold| / |gold|, F1 = 2PR/(P+R); an empty prediction
scores 0; scores are averaged over entities.  With `expand=True`, every
identified type also identifies all its ontology ancestors ("all parent types
on the path from the root ... are also considered as identified").
"""
import re

_SPLIT = re.compile(r"[,，、;；\n]|\band\b|\bor\b|和|及")


_COMMENT = re.compile(r"^(note|notes|therefore|however|this|these|in summary|explanation|based on|"
                      r"here|i |the entity|the types|please|so |thus|it |根据|注意|注：|注:|因此|所以|以上|该实体|这个实体|实体)", re.I)


def _clean_piece(p):
    p = p.strip().strip(".。!！\"'“”‘’`*[]【】<>《》").strip()
    p = re.sub(r"^(type\s*\d+\s*[:=]?\s*)", "", p, flags=re.I)
    return p


def parse_types(text, onto, max_items=30):
    """Extract the type list from an LLM answer.

    Comment lines ("Note that ...", "Based on ...") are ignored; a header line
    ending with ':' is skipped; numbered/bulleted items are one label each;
    parenthetical remarks and ' - explanation' tails are dropped when the
    remaining text matches a type.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    labels = []
    for l in lines:
        l2 = re.sub(r"^([\-\*•]|\d+[\.\)])\s*", "", l).strip()
        if not l2:
            continue
        if re.search(r"[:：]\s*$", l2):
            continue  # header such as "The types are:"
        if ":" in l2 or "：" in l2:
            head, tail = re.split(r"[:：]", l2, maxsplit=1)
            # "Types: A, B" -> tail; "A: definition" -> head when it is a type
            if onto.match(_clean_piece(head)):
                l2 = head
            elif tail.strip():
                if _COMMENT.match(l2) or len(head.split()) > 3 or len(head) > 12:
                    l2 = tail
                else:
                    l2 = head
        elif _COMMENT.match(l2) and len(l2) > 40:
            continue
        for piece in _SPLIT.split(l2):
            p = _clean_piece(piece)
            if not p or len(p) > 80:
                continue
            labels.append(p)
    out = []
    for p in labels:
        m = onto.match(p)
        if not m:
            base = _clean_piece(re.split(r"\s+[-–—]\s+|\s*[\(（]", p)[0])
            m = onto.match(base) if base else None
        if not m and re.fullmatch(r"[A-Za-z]+", p) and re.search(r"[a-z][A-Z]", p):
            m = onto.match(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", p))
        lab = m if m else ("<OOV>" + p)
        if lab not in out:
            out.append(lab)
        if len(out) >= max_items:
            break
    return out


def score_entity(pred, gold, onto, expand=True, drop_oov=False):
    pred = [p for p in pred if not (drop_oov and p.startswith("<OOV>"))]
    P = onto.expand(pred) if expand else set(pred)
    G = onto.expand(gold) if expand else set(gold)
    P -= onto.excluded
    G -= onto.excluded
    if not P or not G:
        return 0.0, 0.0, 0.0
    inter = len(P & G)
    p, r = inter / len(P), inter / len(G)
    f = 0.0 if inter == 0 else 2 * p * r / (p + r)
    return p, r, f


def micro(records, onto, expand=True):
    """Micro-averaged P/R/F1 from pooled true positives over all entities."""
    tp = npred = ngold = 0
    for pred, gold in records:
        P = (onto.expand(pred) if expand else set(pred)) - onto.excluded
        G = (onto.expand(gold) if expand else set(gold)) - onto.excluded
        tp += len(P & G); npred += len(P); ngold += len(G)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def aggregate(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    P = sum(r[0] for r in rows) / n
    R = sum(r[1] for r in rows) / n
    F = sum(r[2] for r in rows) / n
    return {"n": n, "P": P, "R": R, "F1": F, "F1_of_means": (2 * P * R / (P + R)) if P + R else 0.0}
