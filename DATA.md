# Data formats

## Annotations (`data/<theme>/annotations.jsonl`)

One JSON object per document.

| Field | Content |
|---|---|
| `doc_id`, `theme` | Document id (battery `BAT01` to `BAT20`, in DOI order; game `G…`) and theme |
| `source` | Battery: `citation`, `doi` and `url`. Game: `platform` and `url`; for HoYoLAB posts also `post_id` and `date`, and for the other three documents a `note` |
| `topic` (battery), `lang` (game) | Flow-battery or electrode paper; language of the post |
| `passage_size` | Size of our copy of the passage: `chars` and `lines` (battery: one sentence per line) |
| `entities` | One record per typed entity (see below) |

Each entity record holds:
- `mention`: the mention string, as it appears in the passage;
- `offset`: the position, in characters, of the occurrence the code uses in our copy of the passage. That is the first occurrence not directly preceded or followed by an ASCII letter or digit (Chinese characters do not count) or, if there is none, the first occurrence;
- `gold`: the gold types;
- `hard`: whether the entity is the document's hard case (one per document). A hard case is an entity that a model is unlikely to know or whose wording suggests other types;
- `kb_entity` (game, where present): the unit of our knowledge base whose types the gold equals.

**Gold types.** Scoring adds all ancestors of the gold and predicted types, so listing an ancestor does not change a score.
- Battery gold never lists the roots Battery and Electrode, and often lists intermediate types too, for example RedoxFlowBattery, FullFlowBattery and AqueousOrganicFlowBattery.
- Game gold lists the most specific types.

## Corpus (`data/<theme>/corpus.jsonl`)

The input of `python -m orag.run`: the annotation records with the text added as `passage`. The code reads `doc_id`, `passage` and, of each entity, `mention`, `gold` and `hard`; it ignores other fields. See `examples/<theme>/corpus.jsonl`.

## Label ontologies

| File | Content |
|---|---|
| `data/battery/ontology_Battery.txt`, `ontology_Electrode.txt` | One type per line, indented with tabs by depth; a type may appear under several parents |
| `data/game/ontology.txt` | The 73-type game ontology, in the same format |
| `data/game/type_aliases.json` | Aliases of each game type, used to map answers to types |
| `data/game/type_info.json` | English name, aliases and Chinese and English definitions of each game type |

## Knowledge files

You build these yourself (`README.md`, section 2; for the game theme, the last section of this file).

**Battery: `data/battery/battinfo-enriched.txt`.**
- **Format:** one fact per line, grouped by type; empty lines are allowed and any other line is an error.

  | Line | Content |
  |---|---|
  | `<Type> means: <text>` | BattINFO elucidation |
  | `<Type> is also known as <labels>` | Alternative labels, separated by commas |
  | `<Type> includes following types: <A>, <B>, …` | Subtypes |
  | `<Type>: <text>` | The opening of the lead section of the Wikipedia article linked from the type |
  | `<Type>, <property>, <value>` | A Wikidata statement of the item linked from the type |

- **Types:** the types of both label trees, and BattINFO's OrganicCompound with its subclasses.
- **What the code builds:** the ORAG units come from the label-tree types. They hold definitions, aliases, parents, subtypes and filtered statements as sentences, with the Wikipedia text as the semantic part. The RAG memory holds the lines of all types, ordered by type as in BattINFO's `battinfo-inferred.ttl`; the configurations leave out the subtype lines (`"battery_rag": "ttl_nochildren"`). The code reads that file from `data/battery/battinfo-inferred.ttl` or from the path in `ORAG_BATTINFO_TTL`.

**Game: `data/game/units.jsonl` and `data/game/rag_memory.txt`.**
- `units.jsonl` holds one JSON object per line, one per unit, with no blank lines. Its fields:
  - `type`: for a type unit, the ontology type; for an entity unit, the entity's name, made unique and different from every type name;
  - `K`: the knowledge-dense part, a few sentences. Retrieval sees only `K` and `S`, so nicknames must be in `K`;
  - `S`: the semantic part, descriptions; a string, possibly empty;
  - `parents`: the ontology types of an entity unit (empty for type units);
  - `kind`: `type`, `character`, `skill`, `weapon`, `artifact`, `enemy`, `material`, `creation`, `npc`, `place` or `reaction`. Optional fields `en` (English name), `aliases` and `owner` (for skills and creations) are for reference.
- The code uses `type`, `K`, `S` and `parents`. It warns about repeated `type` values, about parents that are not ontology types, and about entity units (units with `parents`) named like a type.
- `rag_memory.txt` is plain text, each unit's `K` followed by its `S`. The code splits it into chunks.

## Examples (`examples/`)

| File | Content |
|---|---|
| `battery/corpus.jsonl` | An invented flow-battery passage with its annotations; not part of the dataset |
| `battery/battinfo-enriched.txt` | The knowledge lines of four flow-battery types: BattINFO's type names, alternative labels and subtypes, Wikidata statements, and placeholders for the BattINFO elucidations and the Wikipedia texts |
| `game/corpus.jsonl` | An invented post with its annotations; not part of the dataset |
| `game/units.jsonl`, `game/rag_memory.txt` | Four units (a type, a character, a creation and a weapon) and their RAG-memory lines; game and wiki text and the character's nicknames and title are replaced by placeholders in angle brackets |

## Rebuilding the game theme

This section gives the details for rebuilding the corpus and the knowledge base of the game theme; `README.md` gives the overall steps.

### Corpus

- **Posts:** each record of `data/game/annotations.jsonl` gives the post's link in `source.url`. Open it in a browser; on HoYoLAB, switch off automatic translation (`lang` gives the post's language).
- **Passages:** the passage is a contiguous part of the post's text, often all of it, and usually without the title. G10's passage starts with its title. G02's is the thread's title, its opening post and the text of its first two replies, without the forum's edit note and the replies' user and date lines. Use `passage_size.chars` and the offsets to find a passage.
- **Availability:** G01 to G03, two news articles and a forum thread, may no longer be online.

### Knowledge base

The knowledge base consists of `data/game/units.jsonl` and `data/game/rag_memory.txt`. We built ours from genshin-db (https://github.com/theBowja/genshin-db) at commit `49a6544a6c6ae36089cb42fa591fc46f01de8bcf`, in Simplified Chinese and English, and from the game's community wiki (https://genshin-impact.fandom.com).

- **Type units:** one per ontology type. `K` gives the type's definitions from `type_info.json`, its parent and its subtypes, up to 30 of its instances, and its aliases. `S` gives an English description of the concept; we used the lead of its wiki page.
- **Entity units:** one per game entity, about 1,980 in all. Ours has 131 characters, 360 normal attacks, elemental skills and bursts, 254 weapons, 63 artifact sets, 347 enemies, 610 materials, 103 summons and other creations, 48 NPCs, 42 places and 18 elemental reactions.
  - `K` states the entity's Chinese and English names, its nicknames (`又称…`), its main attributes (such as rarity, element, weapon type, region and official character roles) and its ontology types. `parents` lists those types.
  - **Nicknames:** we took the characters' nicknames from their pages on BWIKI (https://wiki.biligame.com/ys), a Chinese community wiki, and added the archon names and the Traveler's aliases ourselves.
  - **Types from the game data:**
    - element: the element types of characters;
    - weapon type: the weapon types of characters and of weapons;
    - enemy type: weekly bosses (category 值得铭记的强敌) to 周本首领, other bosses to 世界首领, elites to 精英敌人, the rest to 普通敌人;
    - material category: the material types.
  - **Character roles:** the official character roles are not in genshin-db; the wiki's Character Role page lists them.
    - DPS and On-Field give 站场输出角色, and DPS and Off-Field give 后台输出角色.
    - Support gives 增伤辅助角色, and Survivability gives 生存辅助角色.
    - We also gave 聚怪辅助角色 to 温迪, 砂糖 and 枫原万叶, and 速切输出角色 to 刻晴, 艾尔海森, 雷电将军 and 优菈.
  - **Other entities:** genshin-db has no NPCs or elemental reactions. We listed those, the places and the creations ourselves, and assigned the types of NPCs, places, artifact sets, creations and reactions from their descriptions. The English NPC and place descriptions are sentences from their wiki pages; the Chinese ones are our own summaries of those pages.
  - **`S` texts:** `S` gives the entity's descriptions, some of them shortened:
    - skills to about 260 characters, plus the first sentence in English;
    - materials to 260 characters, enemies to 300 and weapon effects to 220;
    - the talents in a character's unit to their names, with the first sentence of the elemental skill and burst descriptions.
- **RAG memory:** the same text as plain lines, each unit's `K` followed by its `S`.
- **Formats:** see `examples/game/units.jsonl` and `examples/game/rag_memory.txt`.
- **Gold labels:** where an entity has `kb_entity`, its gold types are exactly that entity's types in our knowledge base.
