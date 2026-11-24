"""Theme ontology: a label hierarchy (DAG) read from tab-indented tree files.

A type can appear several times in a tree file (multiple parents); the paper
counts tree nodes (e.g. 298 + 186 = 484 for the battery theme), while scoring
works on unique type names.
"""
import collections
import re
import unicodedata


def camel_to_words(name):
    """RedoxFlowBattery -> 'redox flow battery'; leaves CJK names untouched."""
    if re.search(r"[一-鿿]", name):
        return name
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return s.lower()


_SPELLING = [("aluminium", "aluminum"), ("sulphur", "sulfur"), ("caesium", "cesium")]


def norm_key(text):
    """Normalisation used to match LLM outputs to type names."""
    t = unicodedata.normalize("NFKC", text).lower()
    for a, b in _SPELLING:
        t = t.replace(a, b)
    t = re.sub(r"[\s_\-–—·・/()（）'\"`*]+", "", t)
    return t


_ABBREV = [(r"\bli[\s\-–]?ion\b", "lithium ion"), (r"\bna[\s\-–]?ion\b", "sodium ion"),
           (r"\bk[\s\-–]?ion\b", "potassium ion"), (r"\bzn[\s\-–]?ion\b", "zinc ion"),
           (r"\bmg[\s\-–]?ion\b", "magnesium ion"), (r"\bli[\s\-–]?s\b", "lithium sulfur"),
           (r"\bni[\s\-–]?mh\b", "nickel metal hydride"), (r"\bpb[\s\-–]?acid\b", "lead acid"),
           (r"\bni[\s\-–]?cd\b", "nickel cadmium")]


def surface_variants(text):
    """Conservative surface variants of an English label (singular forms, without a
    trailing 'system'/'cell', expanded standard battery abbreviations)."""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    if re.search(r"[一-鿿]", t):
        return []
    out = []
    for pat, rep in _ABBREV:
        t2 = re.sub(pat, rep, t)
        if t2 != t:
            t = t2
            out.append(t)
    words = re.split(r"(\s+|-)", t)
    sing = []
    for w in words:
        if re.fullmatch(r"[a-z]+ies", w):
            w = w[:-3] + "y"
        elif re.fullmatch(r"[a-z]+(sses|ches|shes|xes)", w):
            w = w[:-2]
        elif re.fullmatch(r"[a-z]{3,}s", w) and not w.endswith("ss"):
            w = w[:-1]
        sing.append(w)
    t = "".join(sing)
    out.append(t)
    for suf in (" system", " cell", " battery system", " technology"):
        if t.endswith(suf):
            out.append(t[: -len(suf)] + (" battery" if suf != " battery system" else " battery"))
            out.append(t[: -len(suf)])
    # "<metal> metal electrode/anode" -> "<metal> electrode" (BattINFO names metal electrodes by the metal)
    m = re.match(r"^(\w+) metal (electrode|anode|cathode|foil)$", t)
    if m:
        out.append(f"{m.group(1)} electrode")
    if t in ("flow battery", "flowbattery"):
        out.append("redox flow battery")
    if t.endswith(" flow battery") and "redox" not in t:
        out.append(t[: -len(" flow battery")] + " redox flow battery")
    if not re.search(r"(battery|electrode|cell)$", t) and len(t.split()) >= 2:
        out += [t + " battery", t + " electrode"]
    if t in ("rechargeable battery", "rechargeable batteries"):
        out.append("secondary battery")
    return out


class Ontology:
    def __init__(self, tree_files, theme_roots_excluded=(), path_mode="dag"):
        """path_mode: "dag" expands a type to all its ancestors over every occurrence;
        "first" uses only the root path of the type's first occurrence in the tree file."""
        self.path_mode = path_mode
        self.first_path = {}
        self.tree_lines = []  # raw label hierarchy (tab-indented), as in the tree files
        self.parents = collections.defaultdict(set)
        self.children = collections.defaultdict(list)
        self.tree_nodes = 0
        self.order = []
        for path in tree_files:
            stack = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    depth = len(line) - len(line.lstrip("\t"))
                    name = line.strip()
                    self.tree_lines.append("\t" * depth + name)
                    stack = stack[:depth]
                    self.first_path.setdefault(name, list(stack))
                    if stack:
                        p = stack[-1]
                        self.parents[name].add(p)
                        if name not in self.children[p]:
                            self.children[p].append(name)
                    else:
                        self.parents.setdefault(name, set())
                    stack.append(name)
                    self.tree_nodes += 1
                    if name not in self.order:
                        self.order.append(name)
        self.types = list(self.order)
        self.type_set = set(self.types)
        self.roots = [t for t in self.types if not self.parents.get(t)]
        # Theme roots (e.g. Battery, Electrode) are trivially true for every
        # entity of the theme and can be excluded from scoring.
        self.excluded = set(theme_roots_excluded)
        self.aliases = {}  # alias key -> type
        self._key2type = {norm_key(t): t for t in self.types}
        self._key2type.update({norm_key(camel_to_words(t)): t for t in self.types})

    def add_alias(self, alias, type_name):
        k = norm_key(alias)
        if k and k not in self._key2type:
            self._key2type[k] = type_name

    def ancestors(self, t):
        if self.path_mode == "first":
            return set(self.first_path.get(t, ()))
        out, st = set(), [t]
        while st:
            x = st.pop()
            for y in self.parents.get(x, ()):
                if y not in out:
                    out.add(y)
                    st.append(y)
        return out

    def depth(self, t):
        ps = self.parents.get(t)
        if not ps:
            return 0
        return 1 + min(self.depth(p) for p in ps)

    def match(self, text, lenient=True):
        """Map a predicted label string to an ontology type, or None.

        lenient: also accept surface variants a human scorer would accept (plurals,
        a trailing "system"/"cell", standard battery abbreviations)."""
        t = self._key2type.get(norm_key(text))
        if t or not lenient:
            return t
        for v in surface_variants(text):
            t = self._key2type.get(norm_key(v))
            if t:
                return t
        return None

    def expand(self, labels):
        """Labels plus all their ontology ancestors (non-ontology labels kept)."""
        out = set(labels)
        for t in labels:
            if t in self.type_set:
                out |= self.ancestors(t)
        return out

    def stats(self):
        return {"tree_nodes": self.tree_nodes, "unique_types": len(self.types),
                "roots": self.roots,
                "multi_parent": sum(1 for t in self.types if len(self.parents.get(t, ())) > 1)}
