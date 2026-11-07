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
4. **Acoustic Pause Markers (`‖` and `‖‖`)**:
   - `‖` indicates a short spoken pause (>= 0.4s) and `‖‖` indicates a long spoken pause (>= 1.0s). Place a period ONLY if the preceding words already form a complete independent sentence, or a comma if the speaker paused mid-sentence. **Do not include `‖` or `‖‖` in the output**.

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
                    "\n# existing_terms_note\n"
                    "### Full-Video & Candidate Reference Terms\n"
                    "Canonical reference terms from the video and domain glossary (use these to resolve shorthands, abbreviations, or ASR misspellings in the current chunk):\n"
                    + "\n".join(prev_lines)
                )

    summary_prompt = f"""
## 1. 角色与目标
你是一名专业术语专家与视频内容分析师，负责从 {src_lang} ASR（自动语音识别）转录稿中提取关键术语，用于 {tgt_lang} 字幕本地化与质量保证。

## 2. 核心任务
1. **主题总结 (`theme`)**：用两句话精炼概括当前文本块的核心主题与讨论要点。
2. **权威术语核对与简称/错词关联**：
   - `# terms_note` 是当前文本块中已匹配出的权威参考术语（格式为 `src:tgt` 或 `alias:tgt (src)`）；`# existing_terms_note` 是整部视频及领域词库中的相关候选完整术语（格式为 `src:tgt`）。
   - 检查原文 `<text>` 中是否使用了 `# terms_note` 或 `# existing_terms_note` 中某条完整术语的 **口语简称、首字母缩写或 ASR 语音/拼写讹误**（例如参考词条为 `Cuirass of the Falling Star`、`Shards of Galanor`、`Heart of Inmost Light`、`One Thousand Voices`、`Star-Eater Scales`、`Spirit of Verity`，而当前 `<text>` 中简称为 `Cuirass`/`Curious`、`Shards`、`HOIL`/`Hoyle`、`1K`、`Star-Eater`、`Verity`）。若有，请将该完整权威术语填入 `src`，官方译名填入 `tgt`，并将当前 `<text>` 中实际出现的简称或 ASR 错词原样填入 `aliases` 数组！
   - 若 `# terms_note` 中的某条一词多义词（如 `Super`, `adds`, `build`）在当前 `<text>` 中仅作为普通日常副词/动词使用（如 `"super complicated"`, `"want to add"`），请忽略该词，不要将其输出到 `terms`。
   - **绝对严禁**：严禁将名词单复数（如 shotgun/shotguns）、名词所有格（`'s`）、动词时态变化（`-ed`, `-ing`）或包含该词的其他独立词组（如把 `"build score"` 当成 `"build"` 的别名、把 `"surge matching"` 当成 `"Surge"` 的别名）填入 `aliases`！`aliases` **仅限指代同一实体的口语简称、缩写或 ASR 拼写/同音识别错误**。
3. **提取新术语**：提取原文 `<text>` 中未在下方列出但属于专有名词、扩展包/副本/活动名、核心系统与界面模块、武器、护甲、技能、游戏/业务机制、地名、人名、首领名、领域行话的新术语，并给出准确的 {tgt_lang} 译名。
   - **优先提取完整复合专有名词**（如 `Seasonal Hub`、`Vanguard Alerts`、`Player Stakes`、`Order System`、`Returner's Gift`），不要拆成孤立的单个普通词。若原文同时使用了该复合专有名词的单字/双字简称，将简称放入 `aliases`。
   - 常用通用词汇（如 system, player, video, update, missions, activities 等字面直译无歧义词）严禁提取。
   - `tgt` 必须为干净无括号、无句号、无引号的译名。

# terms_note
{terms_note}
{existing_terms_note}

## 3. INPUT
<text>
{source_content}
</text>

## 4. Output in only JSON format and no other text
```json
{{
  "theme": "用两句话总结文本块的核心内容与讨论主题",
  "terms": [
    {{
      "src": "canonical {src_lang} term",
      "tgt": "{tgt_lang} 翻译",
      "aliases": ["原文中出现的 ASR 语音/拼写讹误或简称（无则留空数组 []）"]
    }}
  ]
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return summary_prompt


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
   - **Subtitle breathing room**: Insert natural commas (，) at topic-comment or breath boundaries so each clause stays concise (10–16 Chinese characters, never exceeding 18 characters without a comma pause).
   - Strip meaningless spoken fillers (`uh`, `um`, `you know`, `like`) for crisp, punchy subtitles.
3. **Strict Terminology Adherence & Contextual Sense Disambiguation**:
   - Strictly use the official translations in `### 术语库-词典` (formatted as `src:tgt` or `alias:tgt (canonical_src)`) whenever the source line refers to that domain entity, including across plurals, possessives (`'s`), or verb inflections (`-ed`, `-ing`).
   - **Sense check**: If a word in `### 术语库-词典` is used in the sentence purely as an everyday general expression (e.g., `"the number of"` meaning quantity, `"break ground"` meaning make progress, `"recent success"` meaning achievement), translate its natural everyday meaning instead of forcing a glossary term.
   - **No untranslated common words**: Never leave ordinary {SRC_LANGUAGE} words (such as `past`, `boss`, `update`, `patch`, `test`) untranslated in {TARGET_LANGUAGE} sentences; only standard acronyms/abbreviations (`DPS`, `PvE`, `PvP`, `DoT`) or model numbers may remain in Latin letters.
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
    json_format = {}
    for key, value in faithfulness_result.items():
        json_format[str(key)] = {
            "origin": value.get("origin", ""),
            "direct": value.get("direct", ""),
            "reflect": "reflection on the direct translation version (or 'OK' if already accurate and natural)",
            "free": f"<<请逐行重写为极其自然、地道的{TARGET_LANGUAGE}专业字幕，严格保留术语词典译名，且仅输出最终译文>>",
        }
    json_format_str = json.dumps(json_format, indent=4, ensure_ascii=False)

    prompt_expressiveness = f"""
## Role: Elite Linguistic Polisher & Guardian of Fidelity
{style_prompt}
## Task:
Critically review the 'direct' {TARGET_LANGUAGE} translation provided for each {SRC_LANGUAGE} 'origin' line. Your primary responsibilities are twofold:
1. **Guardian of Fidelity**: Ensure the translation is *uncompromisingly faithful* to the deepest meaning, intent, tone, and all nuances of the original {SRC_LANGUAGE} text.
2. **Artisan of Fluency**: Elevate this faithful translation to achieve the highest level of fluency, naturalness, and idiomaticity, making it indistinguishable from expert-level {TARGET_LANGUAGE} discourse.
You will produce a 'free' translation that meets these exacting standards, strictly adhering to terminology and proper noun rules.

## Context:
- Source Language: {SRC_LANGUAGE}
- Target Language: {TARGET_LANGUAGE}
- Input: 'Origin' {SRC_LANGUAGE} lines, their 'direct' (initial faithful) {TARGET_LANGUAGE} translations, and a glossary.
- Goal: To produce a final 'free' translation that is both a perfect semantic match to the original and a masterpiece of natural {TARGET_LANGUAGE} expression, fully aligning with native cognitive patterns and communication habits.

## Core Instructions:
1. **Scrutinize for Absolute Fidelity & Baseline Assessment**:
   - **a. Verify against Origin (Critical Step)**: Meticulously compare the 'origin' ({SRC_LANGUAGE}) text with the 'direct' ({TARGET_LANGUAGE}) translation. Your foremost duty is to verify that the 'direct' translation flawlessly captures the complete meaning, intent, tone, and all subtle nuances of the original. Identify and rectify any loss, distortion, omission, or misrepresentation of meaning in the 'direct' translation before proceeding.
   - **b. Assess Initial Naturalness**: Evaluate adherence to {TARGET_LANGUAGE} grammatical norms, established thinking patterns, reading flow, and auditory conventions.
2. **Uphold Verified Fidelity**:
   - The 'free' translation MUST preserve the full meaning, intent, tone, and nuances of the 'origin' text. No aspect of the original meaning shall be compromised for the sake of fluency or stylistic preference.
3. **Elevate to Expert-Level Naturalness & Idiomaticity**:
   - Align with {TARGET_LANGUAGE} cognitive sequencing, logical flow, and natural discourse structuring.
   - Unpack long post-nominal modifiers and prepositional chains into short, breathable sequential clauses separated by commas (，), keeping each clause concise (10–16 Chinese characters, never exceeding 18 characters without a comma pause).
   - Remove spoken hesitation fillers (`uh`, `um`, `you know`) and awkward translationese or stacked literal calques.
4. **Mandatory Terminology & Proper Noun Preservation**:
   - **Strict Terminology Enforcement**: Strictly enforce the provided glossary and dictionary in `### 术语库-词典` (formatted as `src:tgt` or `alias:tgt (canonical_src)`). Every glossary translation MUST remain intact in 'free' — never abbreviate, paraphrase, or alter official terms during polishing. When terms appear with inflections (plurals, `'s`, `-ed`, `-ing`), maintain their standard canonical translation.
   - **Proper Nouns**: Ensure proper nouns (not in glossary) remain intact or in their standard {TARGET_LANGUAGE} form.
5. **Strict Subtitle Punctuation & No Annotations**:
   - **Allowed punctuation ONLY**: comma (，), period (。), question mark (？), enumeration comma (、), and exclamation mark (！).
   - **Forbidden punctuation**: NEVER output dashes (`——`, `—`, `--`), colons (`：`, `:`), semicolons (`；`, `;`), quotation marks (`“”`, `‘’`, `""`, `''`, `「」`, `『』`), book title marks (`《》`, `〈〉`), or parenthetical notes. Use a comma (，) instead of dashes, colons, or semicolons, and write all term, skill, and work names as plain text without quotes.
   - Do NOT add comments, explanations, or alternative translations in parentheses in the 'free' translation field. Never append original source terms in parentheses (e.g., "静谧狩猎", NOT "静谧狩猎 (Still Hunt)").
6. **Principle of Necessary Enhancement (Keep-if-Flawless)**:
   - If the 'direct' translation already meets the highest standards of fidelity, naturalness, and terminology adherence, state `"OK"` or `"Direct translation (verified) is flawless and meets all expert criteria."` in `"reflect"` and copy the direct translation into `"free"`. Do not make cosmetic changes for personal preference.
   - If changes were made, briefly explain what was enhanced in `"reflect"` (e.g., "Elevated to idiomatic flow", "Corrected terminology", "Adjusted clause structure").

## Input Data (Original lines & Direct Translations):
```json
{json.dumps(faithfulness_result, ensure_ascii=False, indent=2)}
```

## Terminology & Shared Context:
{shared_prompt}

## Output Format:
Provide your refined translations and reflections strictly in the following JSON format. Fill in the `"reflect"` and `"free"` fields for each line number.

```json
{json_format_str}
```

Think step-by-step for each line: analyze, check rules, refine if needed, provide reflection, and format the output correctly.
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
