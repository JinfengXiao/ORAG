"""Dense passage retrieval for vanilla RAG (text chunks) and ORAG (structured
knowledge units with ontology-guided evidence expansion)."""
import itertools

import numpy as np


class Embedder:
    """Sentence embedder; `query_instruction` is prepended to queries for instruction-tuned
    embedding models (e.g. Qwen3-Embedding, multilingual-e5-instruct)."""

    def __init__(self, model_path, device="cuda", batch_size=32, max_len=1024, query_instruction=None):
        from sentence_transformers import SentenceTransformer
        kw = {}
        if "Qwen3-Embedding" in model_path:
            kw = {"model_kwargs": {"torch_dtype": "bfloat16"}, "tokenizer_kwargs": {"padding_side": "left"}}
        self.model = SentenceTransformer(model_path, device=device, **kw)
        self.model.max_seq_length = max_len
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    def encode(self, texts):
        v = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        return v.astype("float32")

    def encode_queries(self, texts):
        if self.query_instruction:
            texts = [f"Instruct: {self.query_instruction}\nQuery: {t}" for t in texts]
        return self.encode(texts)


def _faiss_ip_index(mat):
    import faiss
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


class ChunkRetriever:
    """Vanilla RAG: fixed-size text chunks ranked by cosine similarity."""

    def __init__(self, chunks, embedder, count_tokens):
        self.chunks = chunks
        self.count_tokens = count_tokens
        self.emb = embedder.encode(chunks)
        self.index = _faiss_ip_index(self.emb)

    def retrieve(self, qvec, budget_tokens, max_chunks=None):
        k = len(self.chunks)
        scores, ids = self.index.search(qvec[None, :], k)
        out, used = [], 0
        for s, i in zip(scores[0], ids[0]):
            n = self.count_tokens(self.chunks[i])
            if used + n > budget_tokens:
                break
            out.append({"text": self.chunks[i], "score": float(s), "chunk_id": int(i)})
            used += n
            if max_chunks and len(out) >= max_chunks:
                break
        return out


class UnitRetriever:
    """ORAG retrieval over structured knowledge units.

    Each node n has a knowledge-dense sub-unit K[n] and (optionally) a
    semantic-rich sub-unit S[n].  Steps (paper Sec. 3.3):
      1. rank all sub-units by similarity to the query; the nodes owning the
         top-k sub-units are the starting nodes;
      2. expansion: for each starting node add the K sub-unit of its parent(s)
         and the units of its top-N children (ranked by similarity);
      3. include S[n] only when sim(query, S[n]) > sim(query, K[n]);
      4. merge identical items and fill the evidence budget in rank order.
    """

    def __init__(self, units, parents, children, embedder, count_tokens):
        self.nodes = list(units)
        self.units = units
        self.parents = parents
        self.children = children
        self.count_tokens = count_tokens
        self.sub = []  # (node, kind, text)
        for n in self.nodes:
            if units[n]["K"].strip():
                self.sub.append((n, "K", units[n]["K"]))
            if units[n].get("S", "").strip():
                self.sub.append((n, "S", units[n]["S"]))
        self.emb = embedder.encode([t for _, _, t in self.sub])
        self.index = _faiss_ip_index(self.emb)
        self.pos = {(n, kind): i for i, (n, kind, _) in enumerate(self.sub)}

    def _sims(self, qvec):
        return self.emb @ qvec

    def retrieve(self, qvec, budget_tokens, top_k=10, top_n_children=3, parent=True,
                 s_rule="greater", max_items=None, children_in_k="all", topk_types=False, child_depth=1,
                 channels="joint", top_k_s=None, allowed_nodes=None):
        sims = self._sims(qvec)
        order = np.argsort(-sims)
        if allowed_nodes is not None:  # start nodes only from the routed sub-ontology; expansion unchanged
            order = np.array([i for i in order if self.sub[i][0] in allowed_nodes], dtype=int)

        def sim(n, kind):
            i = self.pos.get((n, kind))
            return None if i is None else float(sims[i])

        s_hits = set()
        if channels == "separate":
            # knowledge and text sub-units are retrieved with DPR separately (a type counts as retrieved if
            # either its knowledge or its text sub-unit is retrieved); start nodes are the
            # owners of the top-k K sub-units and of the top-k_s S sub-units, interleaved by rank
            ks = [self.sub[i][0] for i in order if self.sub[i][1] == "K"][:top_k]
            ss = [self.sub[i][0] for i in order if self.sub[i][1] == "S"][:top_k if top_k_s is None else top_k_s]
            s_hits = set(ss)
            start = []
            for pair in itertools.zip_longest(ks, ss):
                for n in pair:
                    if n is not None and n not in start:
                        start.append(n)
        else:
            # 1. starting nodes: owners of the top-k sub-units, or (topk_types) the top-k distinct types
            #    ranked by their best sub-unit ("top-k entity types ... regarding the rank of similarities
            #    of sub-units", paper Sec. 3.3.1)
            start = []
            for i in (order if topk_types else order[:top_k]):
                n = self.sub[i][0]
                if n not in start:
                    start.append(n)
                if len(start) >= top_k:
                    break

        def node_score(n):
            vals = [v for v in (sim(n, "K"), sim(n, "S")) if v is not None]
            return max(vals) if vals else -1.0

        items, seen_nodes = [], set()

        def add_node(n, role, with_s_rule=True):
            if (n, role) in seen_nodes or n not in self.units:
                return
            seen_nodes.add((n, role))
            sk, ss = sim(n, "K"), sim(n, "S")
            if sk is not None:
                items.append({"node": n, "kind": "K", "role": role, "text": self.units[n]["K"], "score": sk})
            if ss is not None:
                include_s = (sk is None) or (s_rule == "always") or (s_rule == "greater" and ss > sk) \
                    or (role == "start" and n in s_hits)
                if include_s and with_s_rule:
                    items.append({"node": n, "kind": "S", "role": role, "text": self.units[n]["S"], "score": ss})

        # 2./3. expansion around every starting node
        for n in start:
            add_node(n, "start")
            if parent:
                for p in sorted(self.parents.get(n, ()), key=node_score, reverse=True):
                    if p in self.units and self.units[p]["K"].strip():
                        key = (p, "parent")
                        if key not in seen_nodes:
                            seen_nodes.add(key)
                            items.append({"node": p, "kind": "K", "role": "parent",
                                          "text": self.units[p]["K"], "score": sim(p, "K")})
            frontier = [n]
            for _ in range(child_depth):  # depth 1 = the paper's top-N children; >1 follows the best children down
                nxt = []
                for f in frontier:
                    kids = [c for c in self.children.get(f, ()) if c in self.units]
                    kids = sorted(kids, key=node_score, reverse=True)[:top_n_children]
                    for c in kids:
                        add_node(c, "child")
                    nxt += kids
                frontier = nxt

        # optional: list in a K sub-unit only the children that are themselves in the evidence
        if children_in_k != "all":
            present = {it["node"] for it in items}
            for it in items:
                if it["kind"] != "K":
                    continue
                kids = [c for c in self.children.get(it["node"], ()) if c in present]
                lines = []
                for line in it["text"].split("\n"):
                    if " includes following types: " in line or "包括以下类型：" in line or "的实例包括：" in line:
                        if children_in_k == "retrieved" and kids:
                            head = line.split(":")[0] if " includes following types: " in line else None
                            if head is not None:
                                line = f"{head}: {', '.join(kids)}"
                            else:
                                sep = "包括以下类型：" if "包括以下类型：" in line else "的实例包括："
                                line = line.split(sep)[0] + sep + "、".join(kids) + "。"
                        else:
                            continue
                    lines.append(line)
                it["text"] = "\n".join(lines)

        # 4. merge identical items, keep rank order, fill budget
        out, used, texts = [], 0, set()
        for it in items:
            if it["text"] in texts:
                continue
            n_tok = self.count_tokens(it["text"])
            if used + n_tok > budget_tokens:
                continue
            texts.add(it["text"])
            out.append(it)
            used += n_tok
            if max_items and len(out) >= max_items:
                break
        return out, start
