# Ontology-guided retrieval-augmented generation (ORAG) for theme-specific entity typing: code and annotations

This repository contains an implementation of vanilla RAG and ORAG (ontology-guided retrieval-augmented generation) for theme-specific entity typing, and the annotations of two datasets for reproducing the Llama-3-8B-Instruct results of the ORAG paper ([Xiao, Ding et al., COLM 2024](https://openreview.net/forum?id=cKBmZ2PZ6c); Tables 2 and 3):

- **Battery:** 20 passages from research papers on organic electrode materials and redox flow batteries, with 100 typed entities, 20 of them hard cases. The types come from the BattINFO ontology.
- **Game:** 20 documents about the online game Genshin Impact, in Chinese and English: 17 HoYoLAB forum posts, an NGA forum thread and two news articles. They hold 200 typed entities, 20 of them hard cases, typed with a 73-type game ontology.

## Why the texts and knowledge bases are not included

**For data-license compliance**, this repository does not include the source texts or the knowledge bases.
- **Battery articles:** 13 of the 20 do not allow sharing our extracted passages, because they have no open license or a NoDerivatives license. To keep the release uniform, it includes none of the passages; the battery example is an invented passage.
- **Game posts:** the terms of the forum and news sites do not allow reposting their posts.
- **Knowledge bases:** they consist largely of Genshin Impact game data, which HoYoverse owns, and of wiki text under share-alike licenses.

So the repository holds only our code, our annotations and a few small examples. You rebuild the corpora and knowledge bases from the original sources, following the steps below. Your copies will differ from ours in details, such as text extraction, current versions of the game data and wikis, and which posts are still online. We provide no tools for an exact rebuild, so expect your scores to differ somewhat from the paper's.

## Contents

| Path | Content | License |
|---|---|---|
| `orag/` | RAG and ORAG: knowledge units, retrieval, prompts, answer parsing, scoring and comparison with the paper | Apache-2.0 |
| `configs/` | Run configurations: battery RAG, battery ORAG, game RAG and ORAG | Apache-2.0 |
| `scripts/run.sh`, `scripts/score.py` | Run all configurations, then score the answers | Apache-2.0 |
| `data/battery/annotations.jsonl`, `data/game/annotations.jsonl` | Annotations: sources, mentions, gold types and hard cases | Apache-2.0 |
| `data/battery/ontology_Battery.txt`, `ontology_Electrode.txt` | Battery label ontology: BattINFO's Battery and Electrode trees | CC BY 4.0 (BattINFO) |
| `data/game/ontology.txt`, `type_aliases.json`, `type_info.json` | Game label ontology: hierarchy, aliases, English names and definitions | Apache-2.0 |
| `examples/` | Examples of a corpus and of the knowledge files, in the formats the code reads | Apache-2.0, with BattINFO and Wikidata content (see `NOTICE`) |

`DATA.md` describes the formats and the details of rebuilding the game theme. `LICENSE` holds the license, and `NOTICE` the attributions.

## 1. Rebuild the corpora

Each line of `data/<theme>/annotations.jsonl` is one document. Add the document's text as `passage` and save the records as `data/<theme>/corpus.jsonl`; `examples/<theme>/corpus.jsonl` shows the result.

- **Battery:** get the article from the DOI in the record's `source.doi` field. The passage is a run of 7 to 13 consecutive sentences (`passage_size.lines`; we put one sentence per line). Each entity's `offset` gives its position in our copy of the passage, so the smallest offset shows roughly where the passage starts.
- **Game:** open each post at the link in the record's `source.url` field, for example `https://www.hoyolab.com/article/32174703` for G09, and take the passage from its text. `DATA.md` gives the details, including the documents that may no longer be online.
- **Mentions:** every mention must occur verbatim in the passage. The code finds each mention's sentence by its first occurrence that is not next to an ASCII letter or digit (`DATA.md` gives the rule), and `python -m orag.run` warns about mentions it cannot find. Where your text renders a mention differently (spacing, ligatures, sub- and superscripts), edit the passage or the mention, or drop the entity.
- **Missing documents:** leave out documents you cannot rebuild; the scores then cover the rest.

## 2. Build the knowledge bases

**Battery** (`data/battery/battinfo-enriched.txt`):
- **BattINFO:** download `battinfo-inferred.ttl` from BattINFO (https://github.com/BIG-MAP/BattINFO) at commit `6a61af9c7a9f26257788609db31ce094789861df`. Save it as `data/battery/battinfo-inferred.ttl`, or set `ORAG_BATTINFO_TTL` to its path. The battery RAG memory follows the order of this file.
- **Types:** cover every type in the two ontology files, and BattINFO's OrganicCompound with its subclasses (75 classes; they enter only the RAG memory).
- **Lines from BattINFO:** for each type, write its BattINFO elucidation (`<Type> means: ...`), its alternative labels, separated by commas (`<Type> is also known as ...`), and its subtypes (`<Type> includes following types: ...`).
- **Wikipedia:** where BattINFO links the type to Wikipedia, add the opening of the article's lead section, in whole sentences and up to about 1,000 characters (`<Type>: ...`).
- **Wikidata:** where BattINFO links the type to Wikidata, add a selection of the item's statements, with English labels (`<Type>, <property>, <value>`).
  - Ours has about 8 statements per item, over 27 properties: mostly subclass of, has part(s), part of, instance of and Commons category.
  - For compounds it also has chemical structure, chemical formula and database identifiers.
- **Format:** `examples/battery/battinfo-enriched.txt` shows it. The code builds the ORAG units and the RAG memory from this one file.

**Game** (`data/game/units.jsonl` and `data/game/rag_memory.txt`): build one knowledge unit per ontology type and per game entity from genshin-db (https://github.com/theBowja/genshin-db, commit `49a6544a6c6ae36089cb42fa591fc46f01de8bcf`) and the game's community wiki. `DATA.md` describes the units, how entities map to types and how long the texts are, and `examples/game/` shows the formats.

## 3. Run and score

Install the packages in `requirements.txt` on a machine with two CUDA GPUs. By default the LLM runs on the first GPU and the embedders on the second; set `embed_device` in the configurations to change this. `orag/run.py` uses `meta-llama/Meta-Llama-3-8B-Instruct` (accept Meta's license on Hugging Face to download it), `Qwen/Qwen3-Embedding-8B` and `sentence-transformers/all-MiniLM-L6-v2` by default; set `llm_model`, `embed_model` or `minilm_model` in a configuration to change them. Then run:

```
bash scripts/run.sh
python -m orag.compare runs/battery/battery_rag runs/battery/battery_orag runs/game/game
```

- **`run.sh`:** runs battery RAG, battery ORAG and the game configuration, then prints precision, recall and F1 for all entities and for the hard cases.
- **`orag.compare`:** puts the runs' own summaries next to the Llama-3 rows of Tables 2 and 3 of the paper. Its answer mapping can differ slightly from `scripts/score.py`.
- **Scoring:** for each entity, the predicted and gold types are both extended with all their ontology ancestors. For the battery theme the roots Battery and Electrode are left out. P, R and F1 are computed per entity and averaged over entities.
- **Re-runs:** repeated runs can differ slightly, because the embeddings are recomputed on the GPU.

## License

- **This repository** is licensed under the Apache License 2.0 (`LICENSE`): the code, the annotations, the ontology files, the examples and the documentation.
- **Third-party parts** keep their own licenses; `NOTICE` lists them with their attributions:
  - BattINFO content, under CC BY 4.0: the battery ontology files, the BattINFO type names, alternative labels and subtype lists in the battery knowledge example, and a BattINFO definition quoted in `orag/kb.py`;
  - the Wikidata statements in the battery knowledge example, under CC0 1.0.
- **Files you build:** corpora and knowledge bases built from the original sources contain those sources' text, which stays under the sources' own licenses and terms.

## Citation

If you use this code or these annotations, please cite the ORAG paper:

```
@inproceedings{xiao2024orag,
  title     = {{ORAG}: Ontology-Guided Retrieval-Augmented Generation for Theme-Specific Entity Typing},
  author    = {Xiao, Jinfeng and Ding, Linyi and Barry, James and Elkaref, Mohab and De Mel, Geeth and Han, Jiawei},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2024}
}
```
