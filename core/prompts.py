import json
from core.utils import get_source_language, load_key, load_key_or


## ================================================================
# @ _3_1_split_nlp.py
def get_punctuation_prompt(text: str, language: str | None = None):
    lang = language or get_source_language()
    return f"""
## Role
You are a speech transcript punctuation and syntactic sentence boundary specialist in **{lang}**.

## Task
Restore sentence-ending punctuation (`.`, `?`, `!` or `。`, `？`, `！`) and internal commas into the given ASR transcript segment so that every split sentence is **100% syntactically and semantically complete**.

## Rules (Syntactic Completeness First — Length is Secondary)
1. **Strict Word Preservation**: NEVER change, add, delete, or reorder any word, character, or number from the input text. Only insert or fix punctuation marks.
2. **Syntactic Completeness Above All (Never Split Dependent Clauses)**:
   - Every sentence MUST express a complete thought with its main clause intact. Sentence length does NOT matter — a 50–70 word complex sentence is completely fine; NEVER chop a complex sentence in half just because it is long.
   - **Fronted Dependent / Conditional / Temporal Clauses**: When a sentence opens with a subordinate clause (e.g., conditional, temporal, concessive, or causal clauses in {lang}, such as `While...`, `Although...`, `If...`, `When...`), place a **comma** after the dependent clause and keep the following main clause in the **same sentence**. NEVER put a sentence-ending period between a fronted dependent clause and its main clause.
   - **Trailing Subordinate & Relative Clauses**: Keep trailing relative, causal, concessive, or exemplifying clauses attached to their main clause with a comma, NOT a period.
   - **Mid-Sentence Spoken Hesitations**: When a speaker pauses or utters a vocal filler mid-sentence before continuing a list, predicate, or phrase, use commas; NEVER start a new sentence with a dangling phrase after a hesitation filler.
3. **Break True Independent Sentences**:
   - When one complete independent thought finishes and a new independent main clause begins, end the first with `.`, `?`, `!` (or `。`, `？`, `！`).
4. **Acoustic Pause Markers (`‖`)**:
   - The symbol `‖` in the input indicates a spoken acoustic pause (>= 0.4s). Use `‖` as a cue: place a period ONLY if the preceding words already form a complete independent sentence, or a comma if the speaker paused mid-sentence. **Do not include `‖` in the output**.

## INPUT
<transcript>
{text}
</transcript>

## Output in only JSON format and no other text
```json
{{
    "restored": "The exact input transcript with proper punctuation restored."
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()


## ================================================================
# @ _3_2_split_meaning.py & _5_split_sub.py
def get_split_prompt(sentence, num_parts=2, word_limit=20, language=None):
    language = language or get_source_language()
    split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
Insert `[br]` tag(s) to split the given subtitle text into **{num_parts}** roughly equal, semantically natural parts (each under **{word_limit}** words/characters).

1. Split at natural syntactic boundaries (punctuation marks, clause boundaries, or conjunctions/prepositions).
2. Keep the parts balanced in length and never alter, add, or omit any characters from the original text.

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output in only JSON format and no other text
```json
{{
    "split1": "Text with [br] tags at split positions",
    "choice": "1"
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return split_prompt


def get_batch_split_prompt(items: dict[str, dict], language=None):
    language = language or get_source_language()
    input_json = json.dumps(items, ensure_ascii=False, indent=2)
    out_template = {
        k: f"<{language} text {k} with [br] inserted to split into {v.get('num_parts', 2)} balanced parts>"
        for k, v in items.items()
    }
    out_json = json.dumps(out_template, ensure_ascii=False, indent=2)
    return f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
For each numbered subtitle line below, insert `[br]` tag(s) to split its `text` into the requested `num_parts` roughly equal, semantically natural parts.
- Split at natural phrase, conjunction, preposition, or predicate boundaries.
- Keep all parts roughly balanced in length.
- Never alter, add, or omit any words/characters from `text` — only insert `[br]`.

## INPUT
```json
{input_json}
```

## Output in only JSON format and no other text
```json
{out_json}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()


## ================================================================
# @ _4_1_summarize.py
def get_summary_prompt(source_content, custom_terms_json=None):
    src_lang = get_source_language()
    tgt_lang = load_key("target_language")

    # add custom terms note
    terms_note = ""
    if custom_terms_json and custom_terms_json.get("terms"):
        terms_list = []
        for term in custom_terms_json["terms"]:
            aliases = term.get("aliases") or []
            alias_str = (
                f" [aliases: {', '.join(str(a) for a in aliases if str(a).strip())}]"
                if aliases
                else ""
            )
            terms_list.append(
                f"- {term['src']}{alias_str}: {term['tgt']} ({term.get('note', '')})"
            )
        terms_note = (
            "\n### Reference & Existing Terms\n"
            "Exclude exact matches already listed below, but DO extract any ASR spelling/phonetic variants "
            "of these terms that appear in the text by setting `src` to the canonical term, `aliases` to the "
            "exact surface variant(s) in `<text>`, and `tgt` to the existing translation:\n"
            + "\n".join(terms_list)
        )

    summary_prompt = f"""
## Role
You are a video translation expert and terminology curator, specializing in {src_lang} ASR transcript analysis and {tgt_lang} localization.

## Task
For the provided {src_lang} video transcript:
1. Summarize the main topic and core point in two sentences (`theme`).
2. Extract only the essential terminology and ASR variants needed to prevent mistranslation or inconsistency (`terms`).

{terms_note}

## Extraction Rules (Precision & Recall Balance)
1. **What to Extract (Must meet at least one)**:
   - **Proper Nouns & Entities**: Names of people, organizations, brands, products, works, locations, or specialized domain items.
   - **Domain Jargon & Polysemous Terms**: Words that have an everyday meaning but carry a specialized meaning in this video's context (where literal translation would be wrong).
   - **ASR Misrecognitions / Spelling Variants**: Terms in `<text>` that are phonetically misrecognized or misspelled by ASR. Put the **canonical standard spelling** in `src` (when identifiable, otherwise the surface form), list the **exact ASR surface variant(s) appearing in `<text>`** in `aliases`, and provide the accurate `{tgt_lang}` translation in `tgt`.
   - **Technical Acronyms & Code Names**.
2. **Counterfactual Exclusion Test (Strict Filter)**:
   - Before extracting any term, ask: *"If this word is NOT in the glossary and a translator translates it literally, will it be wrong or inconsistent?"*
   - If a literal translation is already unambiguous and accurate (e.g., generic words like `system`, `video`, `performance`, `database`, `algorithm`, `player`, `website`, `update`), **DO NOT extract it**.
3. **Translation Requirement**:
   - Provide the **clean, exact `{tgt_lang}` translation** in `tgt` ready for direct insertion into subtitles (only universally kept acronyms like `AI`, `API`, `CPU`, `GPU` may stay in their original form). **Never** put explanations or parenthetical notes inside `tgt` — put all explanations in `note`.

## INPUT
<text>
{source_content}
</text>

## Output in only JSON format and no other text
```json
{{
  "theme": "Two-sentence video summary",
  "terms": [
    {{
      "src": "canonical {src_lang} term",
      "tgt": "{tgt_lang} translation",
      "aliases": ["exact ASR phonetic/spelling variant in text (if any)"],
      "note": "Brief context or note"
    }}
  ]
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return summary_prompt


def get_consolidate_terms_prompt(
    raw_terms: list[dict], theme: str = "", target_language: str | None = None
):
    src_lang = get_source_language()
    tgt_lang = target_language or load_key("target_language")
    raw_json = json.dumps(raw_terms, ensure_ascii=False, indent=2)
    return f"""
## Role
You are a terminology consolidation and official localization specialist for **{src_lang}** to **{tgt_lang}** video translation.

## Task
Below is a raw terminology list extracted from chunks of a video transcript (Theme: {theme or "General"}).
Some entries may include:
- `kb_candidates`: fuzzy-matched candidates from an authoritative reference glossary (`term-map.json`).
- `source_hint: "kb_scan_candidate"`: reference glossary entries detected directly in the transcript via N-gram scan that chunk extraction may have overlooked.

Consolidate, deduplicate, and refine this list into a unified, accurate glossary:
1. **Authoritative Glossary Alignment (`kb_candidates`)**:
   - When an extracted term or its ASR variant corresponds to an authoritative entry in `kb_candidates`, **adopt the canonical `src` and official `{tgt_lang}` translation `tgt` from `kb_candidates`**.
   - **CRITICAL**: Whenever `src` is normalized to a full canonical name, **keep the exact shorthand, abbreviation, or ASR surface forms that appeared in the transcript inside `aliases`** so text matching can still find them in the transcript.
2. **Verify Direct Glossary Scan Candidates (`source_hint: "kb_scan_candidate"`)**:
   - Entries marked with `source_hint: "kb_scan_candidate"` matched the transcript text against the reference glossary and include a `context_snippet` showing the surrounding words in the transcript.
   - **Check `context_snippet` carefully**:
     - **Keep** the entry ONLY if it is used as a genuine domain term, proper noun, entity, technical mechanism, or specialized item in `context_snippet`.
     - **Discard** the entry if `context_snippet` shows it is just an ordinary conversational phrase or number that coincidentally shares a name with a glossary entry (e.g., `"it's cool"` in `"because it's cool"`, `"second chance"` in `"give this a second chance"`, `"plus one"` in `"10 plus one"`).
3. **Cross-Chunk ASR Variant & Shorthand Unification**:
   - Merge duplicate entries that refer to the same entity under different ASR phonetic spellings, abbreviations, or inflections into a single entry, combining all transcript surface forms into `aliases` and using one consistent `{tgt_lang}` `tgt`.
   - Keep `tgt` as a **clean subtitle-ready term** with zero parenthetical explanations (put any explanations in `note`).
   - If an entry accidentally glues two distinct terms together (e.g., a modifier + an entity name), separate them into their respective canonical entries.
4. **Preserve Domain Coverage & Prune Generic Noise**:
   - Keep all genuine domain terms, proper nouns, specialized items, acronyms, and technical jargon across all chunks.
   - Remove purely generic everyday words where literal translation is already unambiguous.

## Raw Extracted Terms
```json
{raw_json}
```

## Output in only JSON format and no other text
```json
{{
  "terms": [
    {{
      "src": "canonical {src_lang} term",
      "tgt": "accurate unified {tgt_lang} translation",
      "aliases": ["exact transcript surface form or ASR variant (if any)"],
      "note": "concise context note"
    }}
  ]
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
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
    return f"""### Context Information
<previous_content>
{prev_text}
</previous_content>

<subsequent_content>
{after_text}
</subsequent_content>

### Content Summary
{summary_prompt or ""}

### Points to Note
{things_to_note_prompt or ""}"""


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
    # The polish step applies the style when there is one, this step stays a direct translation
    style_prompt = (
        "" if load_key("reflect_translate") else get_translation_style_prompt()
    )
    # Split lines by \n
    line_splits = lines.split("\n")

    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[f"{i}"] = {
            "origin": line,
            "direct": f"direct {TARGET_LANGUAGE} translation {i}.",
        }
    json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)

    src_language = get_source_language()
    prompt_faithfulness = f"""
## Role
You are a senior Netflix subtitle localization specialist and native **{TARGET_LANGUAGE}** writer, fluent in both {src_language} and {TARGET_LANGUAGE}.
Your goal is to produce **final-ready, natural, spoken {TARGET_LANGUAGE} subtitles** in a single pass — combining **100% semantic fidelity (zero alteration, zero fabrication, zero omission)** and strict glossary adherence with idiomatic native syntax.

## Task
Translate the numbered {src_language} subtitle lines into faithful, idiomatic {TARGET_LANGUAGE} line by line.

{shared_prompt}

{style_prompt}<translation_principles>
1. **100% Semantic Fidelity & Accurate Syntactic Attachment (Zero Alteration, Zero Omission)**:
   - Every factual detail, technical mechanism, number, comparison, condition, and speaker stance (`in theory`, `potentially`, `at least`, `technically`, `I know for sure`) MUST be preserved accurately. Never invent facts, alter meaning, or drop substantive information.
   - **Resolve Modifier Attachment Carefully**: When a sentence ends with a numerical measurement or prepositional phrase after a comparison (e.g., `"takes X hits, which is much faster than a standard Y at 1.6 seconds"`), attach the measurement (`1.6 seconds`) to the subject being measured (`takes X hits, taking only 1.6 seconds, which is much faster than a standard Y`), never to the baseline noun `Y`. Translate compound technical modifiers by their real mechanism (e.g., rate/time vs. capacity) rather than surface word-for-word guessing.
   - Use `<previous_content>`, `<subsequent_content>`, and `Content Summary` to resolve polysemous domain terms, pronouns, and omitted subjects according to the video's subject matter.
2. **ASR Homophone & Phonetic Error Recovery**:
   - Source lines come from ASR and contain phonetic typos or speaker self-corrections (e.g., `"X, Y, I believe X, I know for sure doesn't count"`). Recover the speaker's intended word from context and `Points to Note`, and translate the coherent intended statement smoothly without mimicking broken ASR stutters.
3. **Strict Terminology & Clean Proper Nouns**:
   - **Mandatory Glossary Match**: Whenever a term or any of its `also transcribed as` aliases in `Points to Note` appears in a line, you **MUST** use the exact target translation specified in `Points to Note` (never translate a listed proper noun literally).
   - **Clean Output**: Keep unlisted person names, model numbers, or version codes in their original form. **Never** insert parenthetical source text, explanations in brackets, or translator notes into the subtitles.
4. **Idiomatic {TARGET_LANGUAGE} Spoken Flow (Zero Translationese)**:
   - **Strip Spoken Disfluencies**: Completely omit vocal hesitation noises (`uh`, `um`, `er`, `ah`), meaningless discourse fillers (`you know`, `like`, `right?` used as verbal tics, `being like, Okay, uh`), and accidental ASR word repetitions (translate repeated phrases once cleanly unless deliberate emphasis).
   - **Unpack Post-Modifiers**: Restructure trailing relative/adverbial clauses (`that...`, `which is...`, `when it comes to...`, `in order to...`) into natural {TARGET_LANGUAGE} clause order instead of stacking awkward modifier chains or stiff literal calques.
   - **Natural Register**: Match the speaker's natural analytical or conversational tone — fluent, clear, and idiomatic in {TARGET_LANGUAGE}, neither stiffly literal nor overly slangy.
5. **Line-Locked Alignment & Balanced Clause Punctuation**:
   - Maintain strict 1-to-1 line alignment with the numbered input lines; never shift semantic content across line numbers.
   - Use natural {TARGET_LANGUAGE} clause punctuation at genuine clause boundaries, and avoid isolating a short 1–2 word sentence-initial conjunction or discourse adverb with a comma when it can flow naturally into the clause.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
"""
    return prompt_faithfulness.strip()


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    json_format = {
        key: {
            "origin": value.get("origin", ""),
            "direct": value["direct"],
            "reflect": "your reflection on direct translation (or 'OK' if already accurate and natural)",
            "free": "your refined translation (or exact copy of direct when reflect is 'OK')",
        }
        for key, value in faithfulness_result.items()
    }
    json_format = json.dumps(json_format, indent=2, ensure_ascii=False)

    src_language = get_source_language()
    style_prompt = get_translation_style_prompt()
    prompt_expressiveness = f"""
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {TARGET_LANGUAGE} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles.
Evaluate and refine each line ONLY where needed to produce accurate, natural {TARGET_LANGUAGE} subtitles:

1. Check each `direct` translation against its `origin` using the 4-Point SubMQM Diagnostic below.
2. If `direct` is already accurate, faithful to `Points to Note`, and natural in {TARGET_LANGUAGE}, write `"OK"` in `reflect` and copy the **exact full text of `direct`** into `free`.
3. Only when a concrete issue is found, briefly note the fix in `reflect` and provide the refined {TARGET_LANGUAGE} subtitle in `free`.
4. Do not add comments, parenthetical source terms, or explanations in the translation, and never leave `free` empty.

{shared_prompt}

{style_prompt}<Translation Analysis Steps>
Please use a two-step thinking process to handle the text line by line:

1. Direct Translation Reflection (`reflect` — 4-Point SubMQM Diagnostic):
   - **Accuracy & ASR Recovery**: Check against `origin` for mistranslations, unrecovered ASR homophone/phonetic errors, or shifted meaning across line boundaries.
   - **Terminology & Cleanliness**: Verify exact adherence to `Points to Note` and ensure no parenthetical source terms or translator notes were inserted.
   - **Anti-Translationese**: Identify awkward literal calques, stacked modifiers, unnatural passive constructions, or rigid source-language word order that sounds robotic in {TARGET_LANGUAGE}.
   - **Spoken Cadence, Filler Removal & Conciseness**: Omit meaningless spoken hesitation fillers (`uh`, `um`, `er`, `ah`, redundant discourse fillers) and check whether the line reads like natural spoken {TARGET_LANGUAGE} within a concise subtitle length while preserving natural internal clause punctuation.

2. {TARGET_LANGUAGE} Free Translation (`free` — Targeted Refinement & Keep-if-Flawless):
   - **Principle of Necessary Enhancement (Keep-if-Flawless)**: If `direct` already passes all four diagnostic checks and sounds like natural native {TARGET_LANGUAGE} speech, write `"OK"` in `reflect` and copy the **full text of `direct`** unchanged into `free` (never write `"OK"` inside `free`). Do not rewrite or substitute synonyms just for the sake of making changes.
   - **Strict Terminology Preservation**: Every glossary translation from `Points to Note` that was correctly used in `direct` MUST remain intact in `free` — never abbreviate, paraphrase, or drop official terms during polishing.
   - When issues are found in Step 1, rewrite the line into idiomatic, spoken {TARGET_LANGUAGE} that matches the video's register (e.g., conversational for tutorials/vlogs, precise for technical content, formal for documentaries).
   - Keep each line's meaning strictly within its own line number (do not merge or shift content across adjacent lines).
</Translation Analysis Steps>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
"""
    return prompt_expressiveness.strip()


## ================================================================
# @ _5_split_sub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    targ_lang = load_key("target_language")
    src_lang = get_source_language()
    src_splits = src_part.split("\n")
    num_parts = len(src_splits)
    src_part = src_part.replace("\n", " [br] ")
    output_schema = {
        "analysis": "Brief analysis of word order, structure, and semantic correspondence between two subtitles",
        "align": [
            {
                f"src_part_{i + 1}": src_splits[i],
                f"target_part_{i + 1}": f"Corresponding aligned {targ_lang} subtitle part",
            }
            for i in range(num_parts)
        ],
    }
    output_json = json.dumps(output_schema, ensure_ascii=False, indent=4)

    align_prompt = f"""
## Role
You are a Netflix subtitle alignment expert fluent in both {src_lang} and {targ_lang}.

## Task
We have {src_lang} and {targ_lang} original subtitles for a Netflix program, as well as a pre-processed split version of {src_lang} subtitles.
Your task is to create the best splitting scheme for the {targ_lang} subtitles based on this information.

1. Analyze the word order and structural correspondence between {src_lang} and {targ_lang} subtitles
2. Split the {targ_lang} subtitles according to the pre-processed {src_lang} split version
3. Never leave empty lines. If it's difficult to split based on meaning, you may appropriately rewrite the sentences that need to be aligned
4. Do not add comments or explanations in the translation, as the subtitles are for the audience to read

## INPUT
<subtitles>
{src_lang} Original: {json.dumps(src_sub, ensure_ascii=False)}
{targ_lang} Original: {json.dumps(tr_sub, ensure_ascii=False)}
Pre-processed {src_lang} Subtitles ([br] indicates split points): {json.dumps(src_part, ensure_ascii=False)}
</subtitles>

## Output in only JSON format and no other text
```json
{output_json}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return align_prompt


## ================================================================
# @ tts_main.py
def get_correct_text_prompt(text):
    return f"""
## Role
You are a text cleaning expert for TTS (Text-to-Speech) synthesis.

## Task
Clean the given text for speech synthesis while keeping it in its **exact original language**:
1. Keep standard sentence and clause punctuation (e.g., `.`, `,`, `?`, `!`, `。`, `，`, `？`, `！`, `、`) and remove unpronounceable symbols, formatting artifacts, or stray brackets.
2. Preserve 100% of the original language, words, and meaning — never translate or paraphrase.

## INPUT
{text}

## Output in only JSON format and no other text
```json
{{
    "text": "cleaned text in the original language"
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
