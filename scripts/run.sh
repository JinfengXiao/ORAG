#!/bin/bash
# Run and score the configurations on your rebuilt corpora and knowledge bases (README.md, sections 1 to 3).
# Needs a CUDA machine with two GPUs: the LLM runs on the first, the embedders on the second (set "embed_device" in the
# configurations to change this). Each theme is skipped if one of its input files is missing.
set -e
CALLER="$PWD"
cd "$(dirname "$0")/.."
missing() { for f in "$@"; do [ -f "$f" ] || { echo "$f"; return 0; }; done; return 1; }

# ORAG_BATTINFO_TTL may be absolute or relative to the directory you call this script from
TTL="${ORAG_BATTINFO_TTL:-data/battery/battinfo-inferred.ttl}"
case "$TTL" in
  /*) ;;
  *) [ -n "$ORAG_BATTINFO_TTL" ] && TTL="$CALLER/$TTL" ;;
esac
if m=$(missing data/battery/corpus.jsonl data/battery/battinfo-enriched.txt "$TTL"); then
  echo "battery: $m is missing; skipped"
else
  export ORAG_BATTINFO_TTL="$TTL"
  python -m orag.run --theme battery --corpus data/battery/corpus.jsonl --out runs/battery --configs "$(cat configs/battery_rag.json)"
  python -m orag.run --theme battery --corpus data/battery/corpus.jsonl --out runs/battery --configs "$(cat configs/battery_orag.json)"
  python scripts/score.py --theme battery --corpus data/battery/corpus.jsonl \
    --records runs/battery/battery_rag/records.jsonl runs/battery/battery_orag/records.jsonl
fi

if m=$(missing data/game/corpus.jsonl data/game/units.jsonl data/game/rag_memory.txt); then
  echo "game: $m is missing; skipped"
else
  python -m orag.run --theme game --corpus data/game/corpus.jsonl --out runs/game --configs "$(cat configs/game.json)"
  python scripts/score.py --theme game --corpus data/game/corpus.jsonl --records runs/game/game/records.jsonl
fi
