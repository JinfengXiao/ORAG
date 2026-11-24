"""Scoring rule: per-entity P/R/F1 after adding all ontology ancestors (battery roots Battery and Electrode left
out), averaged over entities; F1 is the mean of the per-entity F1 values."""
import json
import os
import sys

from .ontology import Ontology
from .kb import parse_battinfo_enriched


def load_onto(theme, path_mode="dag", excluded=("Battery", "Electrode")):
    R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if theme == "battery":
        o = Ontology([f"{R}/data/battery/ontology_Battery.txt", f"{R}/data/battery/ontology_Electrode.txt"],
                     list(excluded), path_mode=path_mode)
        kf = f"{R}/data/battery/battinfo-enriched.txt"
        if not os.path.exists(kf):
            print("note: data/battery/battinfo-enriched.txt is missing; answers are mapped without the BattINFO aliases", file=sys.stderr)
        info, _ = parse_battinfo_enriched(kf) if os.path.exists(kf) else ({}, [])
        for t, d in info.items():
            if t in o.type_set:
                for a in d["alias"]:
                    for x in a.split(","):
                        o.add_alias(x.strip().strip('"').replace('""@en', ''), t)
    else:
        o = Ontology([f"{R}/data/game/ontology.txt"], path_mode=path_mode)
        for t, al in json.load(open(f"{R}/data/game/type_aliases.json", encoding="utf-8")).items():
            for a in al:
                o.add_alias(a, t)
    return o


SCORING = {"battery": ("dag", ("Battery", "Electrode")), "game": ("dag", ())}
