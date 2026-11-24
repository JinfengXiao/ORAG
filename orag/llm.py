"""Llama-3-8B-Instruct inference through vLLM (greedy decoding)."""

QUERY_TEMPLATE = ("What are the types of the entity {entity} in this passage? If there are multiple "
                  "types, return them all. Format the output as type1, type2, ...\n{passage}")

SYSTEMS = {
    # Task framing for the 8B model (the user turn keeps the paper's query verbatim).
    "typing": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
               "You are given knowledge about the entity types of a theme ontology, followed by a "
               "question about an entity mentioned in a passage. Answer with the names of the ontology "
               "types that the entity itself belongs to (not sub-types it may have, and not other "
               "entities), written exactly as the type names appear in the knowledge. Output only the "
               "comma-separated list of type names, nothing else."),
    "typing_fine": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                    "You are given knowledge about the entity types of a theme ontology, followed by a "
                    "question about an entity mentioned in a passage. Identify the most proper "
                    "fine-grained ontology types of the entity itself (not sub-types it may have, and "
                    "not other entities), written exactly as the type names appear in the knowledge. "
                    "Output only the comma-separated list of type names, nothing else."),
    "typing_strict": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                      "You are given knowledge about the entity types of a theme ontology, followed by a "
                      "question about an entity mentioned in a passage. Decide which ontology types the "
                      "entity itself is an instance of. Only include a type if the passage and the knowledge "
                      "show that the entity belongs to it; do not list sub-types or sibling types that merely "
                      "appear in the knowledge, and do not type other entities. Write the type names exactly "
                      "as they appear in the knowledge, in the same language (do not translate them). Output only the comma-separated list of type names, "
                      "nothing else."),
    "typing_specific": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                        "You are given knowledge about the entity types of a theme ontology, followed by a "
                        "question about an entity mentioned in a passage. Give the most specific ontology "
                        "type(s) that the entity itself belongs to; give more than one type only when the "
                        "entity belongs to several distinct types. Do not list sub-types or sibling types that "
                        "merely appear in the knowledge. Write the type names exactly as they appear in the "
                        "knowledge, in the same language (do not translate them). Output only the comma-separated list of type names, nothing else."),
    "typing_precise": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                       "You are given knowledge about the entity types of a theme ontology, followed by a "
                       "question about an entity mentioned in a passage. Decide which ontology types the "
                       "entity itself is an instance of. Be precise: only include a type if the passage and "
                       "the knowledge show that the entity belongs to it. When you are unsure between several "
                       "sub-types, give their common parent type instead of listing the sub-types. Do not type "
                       "other entities. Write the type names exactly as they appear in the knowledge, in the "
                       "same language (do not translate them). Output only the comma-separated list of type "
                       "names, nothing else."),
    "typing_precise_short": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                             "You are given knowledge about the entity types of a theme ontology, followed by a "
                             "question about an entity mentioned in a passage. Decide which ontology types the "
                             "entity itself is an instance of. Be precise: only include a type if the passage and "
                             "the knowledge show that the entity belongs to it. When you are unsure between "
                             "several sub-types, give their common parent type instead of listing the sub-types. "
                             "Usually one to three types are enough. Do not type other entities. Write the type "
                             "names exactly as they appear in the knowledge, in the same language (do not "
                             "translate them). Output only the comma-separated list of type names, nothing else."),
    "typing_inclusive": ("You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". "
                         "You are given knowledge about the entity types of a theme ontology, followed by a "
                         "question about an entity mentioned in a passage. List every ontology type that the "
                         "entity itself belongs to in this passage, across all applicable branches of the "
                         "ontology (for example its kind or chemistry and its role), as long as the passage and "
                         "the knowledge support it. Do not list sub-types or sibling types that do not apply, "
                         "and do not type other entities. Write the type names exactly as they appear in the "
                         "knowledge, in the same language (do not translate them). Output only the "
                         "comma-separated list of type names, nothing else."),
}

SYSTEMS["typing_convention"] = (
    "You are an expert annotator for fine-grained entity typing on the theme \"{theme}\". You are given "
    "knowledge about the entity types of a theme ontology, followed by a question about an entity mentioned in "
    "a passage. Decide which ontology types the entity itself is an instance of, using the passage and the "
    "knowledge. Parent types are added automatically, so give the most specific types that apply and do not "
    "list their parents. If the mention refers to a technology or a class of materials in general, give only "
    "the type its name denotes; if it refers to the system or electrode that the passage builds or tests, give "
    "the most specific type that facts anywhere in the passage support. For an electrode, also give the role "
    "the passage states (for example cathode, anode, working, counter or reference electrode). Do not type "
    "other entities. Write the type names exactly as they appear in the knowledge, in the same language (do not "
    "translate them). Output only the comma-separated list of type names, nothing else.")

# Top-level structure of the label space (paper Sec. 4.1), optionally appended to the system prompt.
ONTOLOGY_NOTES = {
    "battery": ("The theme ontology has two sub-ontologies, rooted at Battery (batteries, cells and their "
                "kinds) and Electrode (electrodes and their kinds): a battery or cell only has types under "
                "Battery, and an electrode or electrode material only has types under Electrode."),
    "game": ("The top-level types of the ontology are 角色, 创造物, 武器, 圣遗物, 敌人, 材料, 技能, 元素反应 and 地区; "
             "an entity only has types under the one top-level type it belongs to."),
}

# Illustrative answers appended to the system prompt. The battery vanadium answer has the same types as
# 8 battery entities of the dataset, and the game answers those of 11 game entities (4 of them hard cases).
EXAMPLES = {
    "battery": ("Examples of answers: \"sodium-ion batteries\" mentioned generically -> SodiumIonBattery; "
                "\"the vanadium redox flow battery\" -> RedoxFlowBattery, FullFlowBattery, "
                "AqueousMetallicFlowBattery, VanadiumRedoxFlowBattery; \"a platinum wire used as counter "
                "electrode\" -> PlatinumElectrode, CounterElectrode."),
    "game": ("Examples of answers: \"迪卢克\" -> 火属性角色, 双手剑角色, 站场输出角色; \"天空之刃\" -> 单手剑; "
             "\"Hilichurl\" -> 普通敌人."),
}

# Alternative battery examples (used with examples="v2").
EXAMPLES["battery_v2"] = ("Examples of answers: \"flow batteries\" mentioned as a field -> RedoxFlowBattery; "
                          "\"the aqueous alloxazine flow cell\" built in the passage -> AqueousOrganicFlowBattery; "
                          "\"a sodium foil used as counter electrode\" -> SodiumElectrode, CounterElectrode.")

PROMPT_STYLES = {
    "knowledge": "Knowledge:\n{evidence}\n\n{query}",
    "question_first": "{query}\n\nKnowledge:\n{evidence}\n\nTypes of the entity:",
    "knowledge_tail": ("Knowledge:\n{evidence}\n\n{query}\n\nAnswer only with names of types that appear in the "
                       "knowledge."),
    # plain concatenation: evidence, blank line, query
    "plain": "{evidence}\n\n{query}",
    # no retrieval at all (context-only baseline)
    "none": "{query}",
}


def build_query(entity, passage):
    return QUERY_TEMPLATE.format(entity=entity, passage=passage)


def build_prompt(style, query, evidence_items):
    evidence = "\n\n".join(it["text"] for it in evidence_items)
    return PROMPT_STYLES[style].format(evidence=evidence, query=query)


class LLM:
    def __init__(self, model_path, max_model_len=8192, gpu_memory_utilization=0.85, seed=0):
        from vllm import LLM as _V
        self.v = _V(model=model_path, max_model_len=max_model_len, seed=seed,
                    gpu_memory_utilization=gpu_memory_utilization, enforce_eager=False)
        self.tok = self.v.get_tokenizer()

    def count_tokens(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def chat(self, prompts, system=None, max_tokens=128, temperature=0.0, allowed_labels=None, max_labels=8,
             top_p=1.0, seed=0):
        """allowed_labels: if given, decoding is constrained to a comma-separated list of these
        type names (1..max_labels items)."""
        from vllm import SamplingParams
        import re as _re

        def params(labels):
            kw = {}
            if labels:
                from vllm.sampling_params import StructuredOutputsParams
                alt = "|".join(_re.escape(x) for x in sorted(labels, key=len, reverse=True))
                kw["structured_outputs"] = StructuredOutputsParams(regex=f"({alt})(, ({alt})){{0,{max_labels - 1}}}")
            return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, **kw)

        # allowed_labels: one list for all prompts, or one list per prompt (list of lists)
        if allowed_labels and isinstance(allowed_labels[0], (list, tuple)):
            sp = [params(lab) for lab in allowed_labels]
        else:
            sp = params(allowed_labels)
        msgs = []
        for p in prompts:
            m = [{"role": "system", "content": system}] if system else []
            m.append({"role": "user", "content": p})
            msgs.append(m)
        outs = self.v.chat(msgs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
