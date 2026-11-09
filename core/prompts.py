import json
from core.utils import get_source_language, load_key, load_key_or


## ================================================================
# @ _3_1_split_nlp.py
def get_punctuation_prompt(text: str, language: str | None = None):
    lang = language or get_source_language()
    return f"""
## Role: ASR Transcript Punctuation & Sentence Boundary Specialist ({lang})

## Task Description:
Restore sentence-ending punctuation (`.`, `?`, `!` or `。`, `？`, `！`) and internal commas into the raw {lang} ASR transcript below so every sentence is **grammatically complete** and **well-paced with internal commas** for subtitle translation.

## Punctuation Rules:
1. **Strict Word Preservation**:
   - NEVER add, delete, modify, or reorder any word, character, or number from the input transcript. Only insert or fix punctuation marks.
2. **Syntactic Completeness (Never Split Dependent Clauses, Never Fuse Run-On Independent Sentences)**:
   - **Keep dependent clauses attached**: Fronted subordinate clauses (`While...`, `Although...`, `If...`, `When...`, `Since...`, `Because...`) and trailing relative, causal, or participial clauses (`which...`, `that...`, `where...`, `because...`, `especially when...`, `making...`) MUST stay in the same sentence as their main clause, separated by a comma (`,`), NEVER a period.
   - **Break independent statements cleanly**: When a complete independent sentence finishes and a new independent main clause begins (including before spoken transitions like `So`, `Now`, `And then`, `But I can assure you`, `On top of that`, `However` when preceded by a complete clause), end the first sentence with `.`, `?`, or `!` instead of chaining multiple independent sentences into a 50+ word run-on.
3. **Rich Internal Clause Commas (Avoid Long Unpunctuated Runs)**:
   - Insert commas (`,`) at all natural clause and phrase boundaries: after introductory/fronted phrases, before coordinate verb phrases (`...when holding down the trigger, and subtracted it...`), before trailing prepositional/participial phrases (`..., on top of removing...`), and around parenthetical remarks or spoken hesitations.
   - Do not leave a 20+ word stretch without an internal comma when a natural phrase or clause boundary exists.
4. **Acoustic Pause Markers (`‖` and `‖‖`)**:
   - `‖` (>= 0.4s pause) and `‖‖` (>= 1.0s pause) mark spoken pauses: place a period ONLY if the preceding words form a complete independent sentence, or a comma if the pause falls at a clause/phrase boundary. **Do not include `‖` or `‖‖` in your output.**

## Input Transcript:
<transcript>
{text}
</transcript>

## Output Format:
Return ONLY the following JSON object:
```json
{{
    "restored": "The exact input transcript with proper punctuation restored."
}}
```
""".strip()


## ================================================================
# @ _3_2_split_meaning.py & _5_split_sub.py
def get_split_prompt(sentence, num_parts=2, word_limit=20, language=None):
    language = language or get_source_language()
    num_br = max(1, int(num_parts) - 1)
    split_prompt = f"""
## Role: Subtitle Line-Splitting Specialist in **{language}**

## Task Description:
Insert **exactly {num_br}** `[br]` tag(s) into the given subtitle sentence to divide it into **exactly {num_parts}** semantically natural, roughly balanced subtitle segments (each ideally under **{word_limit}** words/characters).

## Splitting Rules:
1. **Strict Character Preservation**: NEVER alter, add, or delete any word, character, or punctuation mark from the original sentence—only insert `[br]`.
2. **Syntactic Integrity (Where to Split vs. Never Split)**:
   - **Preferred split points**: existing punctuation, clause boundaries, topic-comment transitions, after enclosing temporal/locative frames (`在……之前/之后/时/中`), or before conjunctions/prepositional phrases.
   - **Forbidden split points**: NEVER split inside a proper noun, equipment/skill/entity name, or fixed compound term; NEVER split between a quantifier/modifier/`的` and its head noun, inside an enclosing `在……之前/中` frame, between a preposition (`在`, `把`, `对`, `用`, `with`, `of`, `for`, `to`) and its object, or between an auxiliary/modal verb and its main verb.
3. **Balanced Length**: Keep the {num_parts} segments balanced in length; never produce a tiny 1–3 character/word orphan fragment.

## Input Sentence:
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output Format:
Return ONLY the following JSON object:
```json
{{
    "split": "Exact input text with {num_br} [br] tag(s) inserted"
}}
```
""".strip()
    return split_prompt


def get_batch_split_prompt(items: dict[str, dict], language=None):
    language = language or get_source_language()
    input_json = json.dumps(items, ensure_ascii=False, indent=2)
    out_template = {
        k: f"<{language} text {k} with exactly {max(1, int(v.get('num_parts', 2)) - 1)} [br] inserted to split into {v.get('num_parts', 2)} balanced parts>"
        for k, v in items.items()
    }
    out_json = json.dumps(out_template, ensure_ascii=False, indent=2)
    return f"""
## Role: Subtitle Line-Splitting Specialist in **{language}**

## Task Description:
For each numbered subtitle entry in `INPUT`, insert **exactly `num_parts - 1`** `[br]` tag(s) into its `text` to divide it into **exactly `num_parts`** semantically natural, roughly balanced subtitle segments.

## Splitting Rules:
1. **Strict Character Preservation**: NEVER alter, add, or delete any word, character, or punctuation mark from `text`—only insert `[br]`.
2. **Syntactic Integrity (Where to Split vs. Never Split)**:
   - **Preferred split points**: clause boundaries, topic-comment transitions, predicate boundaries, after enclosing temporal/locative frames (`在……之前/之后/时/中`), or before conjunctions/prepositional phrases.
   - **Forbidden split points**: NEVER split inside a proper noun, equipment/skill/entity name, or fixed compound term; NEVER split between a quantifier/modifier/`的` and its head noun, inside an enclosing `在……之前/中` frame, between a preposition (`在`, `把`, `对`, `用`, `向`, `with`, `of`, `for`, `to`) and its object, or between an auxiliary/modal verb and its main verb.
3. **Balanced Length**: Keep all `num_parts` segments balanced in length; never produce a tiny 1–3 character/word orphan fragment.

## INPUT:
```json
{input_json}
```

## Output Format:
Return ONLY the following JSON object:
```json
{out_json}
```
""".strip()


## ================================================================
# @ _4_1_summarize.py
def _format_term_prompt_line(term: dict, canonical_only: bool = False) -> str:
    src = str(term.get("src") or "").strip()
    tgt = str(term.get("tgt") or "").strip()
    if not src or not tgt:
        return ""
    if canonical_only:
        return f"{src}:{tgt}"
    aliases = [
        str(a).strip()
        for a in (term.get("aliases") or [])
        if str(a).strip() and str(a).strip().lower() != src.lower()
    ]
    src_matched = bool(term.get("src_matched", not aliases))
    lines = [
        *([f"{src}:{tgt}"] if src_matched else []),
        *(f"{al}:{tgt} ({src})" for al in aliases),
    ]
    return "\n".join(lines)


def get_summary_prompt(
    source_content, custom_terms_json=None, previously_extracted_terms=None
):
    src_lang = get_source_language()
    tgt_lang = load_key("target_language")

    terms_list = []
    if (
        custom_terms_json
        and isinstance(custom_terms_json, dict)
        and custom_terms_json.get("terms")
    ):
        for term in custom_terms_json["terms"]:
            if not isinstance(term, dict):
                continue
            line = _format_term_prompt_line(term)
            if line:
                terms_list.append(line)

    terms_note = "\n".join(terms_list) if terms_list else "None"

    existing_terms_note = ""
    if previously_extracted_terms:
        prev_terms = (
            previously_extracted_terms.get("terms", [])
            if isinstance(previously_extracted_terms, dict)
            else previously_extracted_terms
        )
        if isinstance(prev_terms, list) and prev_terms:
            prev_lines = [
                _format_term_prompt_line(t, canonical_only=True)
                for t in prev_terms
                if isinstance(t, dict) and str(t.get("src") or "").strip()
            ]
            prev_lines = [line for line in prev_lines if line]
            if prev_lines:
                existing_terms_note = (
                    "\n<video_reference_terms>\n"
                    + "\n".join(prev_lines)
                    + "\n</video_reference_terms>"
                )

    summary_prompt = f"""
## Role: Senior Terminology & ASR Transcript Curator

## Task Description:
Analyze the provided {src_lang} ASR video transcript chunk (`<text>`) for {tgt_lang} subtitle localization:
1. Summarize the core topic and discussion points of `<text>` in two concise sentences (`theme`).
2. Extract **all genuine domain terms, spoken shorthands, and ASR-corrupted entity spans present in `<text>`** into `terms`, using `<chunk_candidate_terms>` and `<video_reference_terms>` to guarantee accurate official translations and canonical forms. Prioritize **zero omission of real domain terms first**, and **zero false-positive noise second**.

## Terminology Extraction Rules:
1. **Priority 1 — Confirm Genuine Candidates**:
   - `<chunk_candidate_terms>` lists domain glossary entries pre-scanned from `<text>` (formatted as `src:tgt` or `alias:tgt (canonical_src)`); `<video_reference_terms>` lists canonical domain terms detected across the full video (`src:tgt`).
   - Whenever `<text>` refers to a candidate term's domain concept—even if lowercase in ASR, or expressed via a spoken shorthand, acronym, or ASR phonetic/spelling error—you **MUST** extract it into `terms` using the exact canonical `src` (when formatted as `alias:tgt (canonical_src)`, always put `canonical_src` into `src` and `alias` into `aliases`) and official `tgt` from the reference list.
2. **Priority 2 — Actively Detect Unlisted Domain Entities & Phonetic ASR Mishearings from Sentence Syntax**:
   - Spoken ASR transcripts frequently mishear proper nouns, equipment, perks/traits, abilities, bosses, UI modules, or mechanics into phonetically similar words, split compounds, or ordinary words that make no grammatical sense in an entity/modifier slot.
   - Whenever any word or phrase in `<text>` occupies a domain entity slot or is a spoken shorthand / phonetic mishearing not covered by `<chunk_candidate_terms>`:
     - Put the reconstructed canonical {src_lang} term in `src` (if you recognize the true canonical name from domain context, write the canonical name in `src`; if unsure, write the transcript phrase in `src`).
     - Put **every exact raw ASR spelling or shorthand appearing in `<text>`** inside `aliases`.
3. **Priority 3 — Reject Coincidental Everyday Words & Keep `aliases` Clean**:
   - **Reject coincidental everyday homophones**: Text scanning may match candidate entries whose words appear in `<text>` purely as ordinary everyday expressions rather than the domain entity (e.g., `"from the title"` meaning video title, `"as well as"` / `"pretty well"`, `"adds 120 DPS"` / `"want to add"` as a verb, `"for the same purpose"`, `"the number of"`, `"in general"`, `"the other half"`, `"double your..."`, `"super complicated"`, `"by process of elimination"`, `"meeting certain conditions"`). If a candidate entry is NOT used in its domain-specific sense in `<text>`, **DO NOT extract it**.
   - **No generic everyday words**: Do NOT extract ordinary common nouns/verbs (`mission`, `story`, `system`, `player`, `video`, `update`, `activity`, `strategy`, `option`, `test`, `phase`, `screen`) whose literal translation is already natural and unambiguous.
   - `aliases` is ONLY for the exact **spoken shorthand, acronym, or ASR transcription error** that replaces `src` in `<text>` (leave `aliases: []` if `<text>` uses the canonical `src` or normal inflections; never include surrounding sentence modifiers or normal plural/possessive/verb inflections).
   - `tgt` must be a clean, final {tgt_lang} translation without parentheses, explanations, quotes, or punctuation.

## Injected Domain Glossary Candidates:
<chunk_candidate_terms>
{terms_note}
</chunk_candidate_terms>{existing_terms_note}

## Input Transcript Chunk:
<text>
{source_content}
</text>

## Output Format:
Return ONLY the following JSON object:
```json
{{
  "theme": "用两句话精炼概括当前文本块的核心主题与讨论要点",
  "terms": [
    {{
      "src": "canonical {src_lang} term",
      "tgt": "clean {tgt_lang} translation",
      "aliases": ["exact shorthand or ASR variant appearing in <text>, or [] if none"]
    }}
  ]
}}
```
""".strip()
    return summary_prompt


def get_kb_disambiguation_prompt(payload: dict) -> str:
    return f"""
## Role: Domain Glossary Grounding Specialist

For each extracted ASR span and its sentence context below, select the matching canonical `src` from `kb_candidates` if and only if the sentence genuinely refers to that glossary item (accounting for ASR phonetic mishearings or spoken shorthands). If none of `kb_candidates` is the intended term in that sentence, output `"NONE"`.

## Input Items:
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## Output Format:
Return ONLY a JSON object mapping each item number (`"1"`, `"2"`, ...) to the exact chosen canonical `"src"` string from `kb_candidates` or `"NONE"`:
```json
{{
  "1": "Chosen Canonical Src or NONE"
}}
```
""".strip()


## ================================================================
# @ _4_2_translate.py & translate_lines.py
def generate_shared_prompt(
    previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt
):
    prev_text = (
        "\n".join(previous_content_prompt)
        if isinstance(previous_content_prompt, list)
        else (previous_content_prompt or "")
    )
    after_text = (
        "\n".join(after_content_prompt)
        if isinstance(after_content_prompt, list)
        else (after_content_prompt or "")
    )
    summary_line = f"Summary: {summary_prompt}\n" if summary_prompt else ""
    return f"""### Context Information
<previous_context>
{prev_text}
</previous_context>

<subsequent_context>
{after_text}
</subsequent_context>

### 术语库-词典
{summary_line}<terminology>
{things_to_note_prompt or ""}
</terminology>""".strip()


def get_translation_style_prompt():
    """The style the user asks for, as a section of the translation prompt; empty without one."""
    style = str(load_key_or("translation_style", "") or "").strip()
    if not style:
        return ""
    return f"""### Translation Style
The user asks for the following style. Follow it in the translation, without changing the meaning of the original:
<translation_style>
{style}
</translation_style>

"""


def get_prompt_faithfulness(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    SRC_LANGUAGE = get_source_language()
    style_prompt = (
        "" if load_key("reflect_translate") else get_translation_style_prompt()
    )
    line_splits = lines.split("\n")

    json_dict = {
        str(i): {
            "origin": line,
            "trans": f"<<请一步到位翻译为精准、自然地道、符合口语字幕节奏的{TARGET_LANGUAGE}最终译文>>",
        }
        for i, line in enumerate(line_splits, 1)
    }
    json_format = json.dumps(json_dict, indent=4, ensure_ascii=False)

    prompt_faithfulness = f"""
## Role: Senior Subtitle Localization Specialist
{style_prompt}
## Task Description:
Translate the provided {SRC_LANGUAGE} video transcript lines into final, broadcast-ready {TARGET_LANGUAGE} subtitles line by line in a **single pass**. Achieve both **uncompromising semantic accuracy** and **native, idiomatic spoken fluency** directly in your output—do not produce stiff literal translationese.

## Translation Guidelines:
1. **Line-by-Line Completeness & Context Continuity**:
   - Translate each line accurately while maintaining smooth logical and tonal continuity with `<previous_context>`, `<subsequent_context>`, and adjacent lines.
   - Preserve all numbers, percentages, mechanics, and causal relationships without omission.
2. **Native Spoken Flow & Clause Unpacking (Single-Pass Polish)**:
   - Organize clauses according to natural {TARGET_LANGUAGE} spoken word order and narrative flow rather than mirroring {SRC_LANGUAGE} syntax.
   - **Unpack long modifiers**: Break up post-nominal clauses (`that ...`, `which ...`, `with ...`, `where ...`) and stacked modifiers into sequential short clauses separated by commas (，). Avoid long `的` chains or clunky `在……上/中` wrappers.
   - **Modifier & equipment roll order**: When stacked attributes, perks, or traits precede an equipment or entity noun (e.g., `[Perk A] [Perk B] [Weapon Name]`), connect the attributes as modifiers of the head noun in natural {TARGET_LANGUAGE} order rather than outputting a flat, disconnected list of nouns.
   - **Subtitle breathing room**: Insert natural commas (，) at topic-comment or breath boundaries so each clause stays concise (10–16 Chinese characters, never exceeding 18 characters without a comma pause).
   - Strip meaningless spoken fillers (`uh`, `um`, `you know`, `like`) for crisp, punchy subtitles.
3. **Strict Terminology Adherence & Contextual Sense Disambiguation**:
   - Strictly use the official translations in `### 术语库-词典` (formatted as `src:tgt`, `alias:tgt (canonical_src)`, or with `[note]`) whenever the source line refers to that domain entity, including across plurals, possessives (`'s`), or verb inflections (`-ed`, `-ing`).
   - **Sense check**: If a word in `### 术语库-词典` is used in the sentence purely as an everyday general expression (e.g., `"the number of"` meaning quantity, `"break ground"` meaning make progress, `"recent success"` meaning achievement), translate its natural everyday meaning instead of forcing a glossary term.
   - **Contextual ASR homophone recovery**: Spoken ASR transcripts frequently contain phonetic mishearings, split compounds, truncated suffixes, or portmanteaus (such as `-less` meaning "without", homophones conflicting with sentence grammar, or corrupted proper nouns). Infer the speaker's intended meaning from surrounding context and `### 术语库-词典` rather than translating corrupted literal words verbatim.
   - **No untranslated common words**: Never leave ordinary {SRC_LANGUAGE} words (such as `past`, `boss`, `update`, `patch`, `test`, `inevitably`) untranslated in {TARGET_LANGUAGE} sentences; only standard acronyms/abbreviations (`DPS`, `PvE`, `PvP`, `DoT`) or model numbers may remain in Latin letters.
4. **Strict Subtitle Punctuation & Zero Annotations**:
   - **Allowed punctuation ONLY**: comma (，), period (。), question mark (？), enumeration comma (、), and exclamation mark (！).
   - **Forbidden punctuation**: NEVER output dashes (`——`, `—`, `--`), colons (`：`, `:`), semicolons (`；`, `;`), quotation marks (`“”`, `‘’`, `""`, `''`, `「」`, `『』`), book title marks (`《》`, `〈〉`), or parenthetical notes. Use a comma (，) instead of dashes, colons, or semicolons, and write all term, skill, and work names as plain text without quotes.
   - Do NOT append original source terms or explanations in parentheses (write `静谧狩猎`, NEVER `静谧狩猎 (Still Hunt)`).

## Input Text Lines:
<source_lines>
{lines}
</source_lines>

## Terminology & Shared Context:
{shared_prompt}

## Output Format:
Return ONLY the following JSON object, filling `"trans"` with the final polished {TARGET_LANGUAGE} subtitle for each line:
```json
{json_format}
```
""".strip()
    return prompt_faithfulness


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    SRC_LANGUAGE = get_source_language()
    style_prompt = get_translation_style_prompt()
    input_clean = {}
    json_format = {}
    for key, value in faithfulness_result.items():
        k = str(key)
        input_clean[k] = {
            "origin": value.get("origin", ""),
            "trans": value.get("trans") or value.get("direct", ""),
        }
        json_format[k] = {
            "reflect": "brief check for omissions, terminology errors, or stiff syntax (or 'OK' if already accurate and natural)",
            "free": f"final polished {TARGET_LANGUAGE} subtitle (or 'OK' if reflect is 'OK')",
        }
    json_format_str = json.dumps(json_format, indent=4, ensure_ascii=False)

    prompt_expressiveness = f"""
## Role: Senior Subtitle Quality Auditor & Polisher
{style_prompt}
## Task Description:
Review each {SRC_LANGUAGE} source line (`origin`) and its initial {TARGET_LANGUAGE} subtitle translation (`trans`), and output the final broadcast-ready subtitle in `free`:
1. **Principle of Necessary Enhancement (Keep-if-Flawless)**:
   - If `trans` is already semantically complete, accurate to `### 术语库-词典`, naturally phrased in spoken {TARGET_LANGUAGE}, and properly punctuated, set `"reflect": "OK"` and `"free": "OK"` (or copy `trans` unchanged). Do NOT rewrite good lines just to change wording.
2. **Targeted Fixes Only**:
   - **Semantic fidelity**: Fix any omission, number/percentage error, or distorted causal relationship against `origin`.
   - **Strict terminology enforcement**: Enforce every official term in `### 术语库-词典` (formatted as `src:tgt` or `alias:tgt (canonical_src)`), including across plurals, possessives (`'s`), and verb inflections (`-ed`, `-ing`). If a glossary word was forced onto a coincidental everyday expression, restore its natural everyday meaning.
   - **Spoken flow & clause breathing room**: Unpack stacked modifiers (`的` chains) or stiff translationese into natural sequential clauses separated by commas (，), keeping each clause concise (10–16 Chinese characters, never exceeding 18 characters without a comma pause).
3. **Strict Subtitle Punctuation & Zero Annotations**:
   - **Allowed punctuation ONLY**: comma (，), period (。), question mark (？), enumeration comma (、), and exclamation mark (！).
   - **Forbidden punctuation**: NEVER output dashes (`——`, `—`, `--`), colons (`：`, `:`), semicolons (`；`, `;`), quotation marks (`“”`, `‘’`, `""`, `''`, `「」`, `『』`), book title marks (`《》`, `〈〉`), or parenthetical notes.

## Input Lines & Initial Translations:
```json
{json.dumps(input_clean, ensure_ascii=False, indent=2)}
```

## Terminology & Shared Context:
{shared_prompt}

## Output Format:
Return ONLY the following JSON object:
```json
{json_format_str}
```
""".strip()
    return prompt_expressiveness


## ================================================================
# @ _5_split_sub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    targ_lang = load_key("target_language")
    src_lang = get_source_language()
    src_splits = src_part.split("\n")
    num_parts = len(src_splits)
    src_part = src_part.replace("\n", " [br] ")
    output_schema = {
        "analysis": f"Brief syntactic mapping between the {num_parts} {src_lang} segments and {targ_lang} clauses",
        "align": [
            {
                f"src_part_{i + 1}": src_splits[i],
                f"target_part_{i + 1}": f"Corresponding aligned {targ_lang} subtitle part {i + 1}",
            }
            for i in range(num_parts)
        ],
    }
    output_json = json.dumps(output_schema, ensure_ascii=False, indent=4)

    align_prompt = f"""
## Role: Bilingual Subtitle Alignment Specialist ({src_lang} -> {targ_lang})

## Task Description:
Partition the `{targ_lang}` subtitle into **{num_parts}** non-empty segments (`target_part_1` ... `target_part_{num_parts}`) so each segment semantically corresponds to its paired `{src_lang}` segment (`src_part_1` ... `src_part_{num_parts}`).

## Alignment Rules:
1. **Semantic & Syntactic Correspondence**: Match each `target_part_i` to the meaning spoken during `src_part_i`. When `{src_lang}` and `{targ_lang}` have inverted clause or modifier order, minimally reorder the `{targ_lang}` clauses so each segment matches what is being said in `src_part_i`.
2. **100% Terminology & Meaning Preservation**: Preserve all domain terms, numbers, and core meaning from `{targ_lang} Original`—never omit content or add parenthetical explanations.
3. **Non-Empty & Clean Subtitle Punctuation**: Every `target_part_i` MUST be non-empty and end at a natural syntactic boundary without forbidden punctuation (`——`, `：`, `；`, quotes, or parentheses).

## INPUT:
<subtitles>
{src_lang} Original: {json.dumps(src_sub, ensure_ascii=False)}
{targ_lang} Original: {json.dumps(tr_sub, ensure_ascii=False)}
Pre-processed {src_lang} Subtitles ([br] indicates split points): {json.dumps(src_part, ensure_ascii=False)}
</subtitles>

## Output Format:
Return ONLY the following JSON object:
```json
{output_json}
```
""".strip()
    return align_prompt


## ================================================================
# @ tts_main.py
def get_correct_text_prompt(text):
    return f"""
## Role: Text Cleaning Specialist for TTS (Text-to-Speech) Synthesis

## Task Description:
Clean `<input_text>` so a TTS engine can synthesize it smoothly in its **exact original language**:
1. Strip non-spoken stage/audio tags (such as `[♪ Music ♪]`, `[Applause]`, `【注意】`), emojis, markdown/HTML artifacts, dashes (`—`, `——`), and stray brackets.
2. Keep only natural spoken words and standard sentence/clause pause punctuation (`.`, `,`, `?`, `!`, `。`, `，`, `？`, `！`, `、`).
3. Preserve 100% of the original language, words, numbers, and meaning—never translate or paraphrase.

## INPUT:
<input_text>
{text}
</input_text>

## Output Format:
Return ONLY the following JSON object:
```json
{{
    "text": "cleaned speakable text in the original language"
}}
```
""".strip()
