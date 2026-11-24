"""Compare run summaries with the paper's LLAMA-3 rows (Tables 2 and 3).

usage: python -m orag.compare runs/<battery_run>/<cfg> [runs/<game_run>/<cfg>] [--flat]
"""
import json
import sys

# (precision, recall, F1) for Llama-3-8B-Instruct, paper Tables 2 (all) and 3 (hard)
TARGETS = {
    ("battery", "all", "rag"): (0.86, 0.56, 0.68), ("battery", "all", "orag"): (0.97, 0.80, 0.83),
    ("game", "all", "rag"): (0.76, 0.61, 0.67), ("game", "all", "orag"): (0.85, 0.80, 0.83),
    ("battery", "hard", "rag"): (0.74, 0.48, 0.53), ("battery", "hard", "orag"): (0.95, 0.76, 0.83),
    ("game", "hard", "rag"): (0.60, 0.56, 0.51), ("game", "hard", "orag"): (0.87, 0.82, 0.83),
}


def rows(summary_path, theme, flat=False):
    s = json.load(open(summary_path))["summary"]
    out = []
    for subset in ("all", "hard"):
        for m in ("rag", "orag"):
            if m not in s:
                continue
            a = s[m][subset + ("_flat" if flat else "")]
            got = (a["P"], a["R"], a["F1"])
            tgt = TARGETS[(theme, subset, m)]
            out.append((theme, subset, m, got, tgt, a["n"]))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flat = "--flat" in sys.argv
    allrows = []
    for p in args:
        cfg = json.load(open(p + "/summary.json"))
        theme = cfg.get("theme") or ("battery" if "battery" in p else "game")
        allrows += rows(p + "/summary.json", theme, flat)
    devs = []
    print(f"{'theme':8s} {'subset':5s} {'method':5s} {'n':>4s}   P (paper)     R (paper)     F1 (paper)")
    for theme, subset, m, got, tgt, n in allrows:
        d = [g - t for g, t in zip(got, tgt)]
        devs += [abs(x) for x in d]
        cells = "  ".join(f"{g:.3f} ({t:.2f})" for g, t in zip(got, tgt))
        print(f"{theme:8s} {subset:5s} {m:5s} {n:4d}   {cells}   max|d|={max(abs(x) for x in d):.3f}")
    if devs:
        print(f"within 0.02: {sum(x <= 0.02 for x in devs)}/{len(devs)}   within 0.025: "
              f"{sum(x <= 0.025 for x in devs)}/{len(devs)}   max |d| = {max(devs):.3f}")


if __name__ == "__main__":
    main()
