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
            src = str(term.get("src") or "").strip()
            if not src:
                continue
            tgt = str(term.get("tgt") or "").strip()
            note = str(term.get("note") or "").strip()
            aliases = term.get("aliases") or []
            alias_str = (
                f" [aliases: {', '.join(str(a) for a in aliases if str(a).strip())}]"
                if aliases
                else ""
            )
            note_str = f" ({note})" if note else ""
            terms_list.append(f"- {src}{alias_str}: {tgt}{note_str}")

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
                f"- {str(t.get('src') or '').strip()}: {str(t.get('tgt') or '').strip()}"
                + (f" ({str(t.get('note') or '').strip()})" if t.get("note") else "")
                for t in prev_terms
                if isinstance(t, dict) and str(t.get("src") or "").strip()
            ]
            if prev_lines:
                existing_terms_note = (
                    "\n# existing_terms_note\n"
                    "### Previously Extracted Terms\n"
                    "Please refer to these terms to maintain consistency across chunks and avoid duplicate entries:\n"
                    + "\n".join(prev_lines)
                )

    summary_prompt = f"""
## 1. 角色与目标
你是一名专业术语专家与视频内容分析师，负责从 {src_lang} ASR（自动语音识别）转录稿中提取关键术语，用于 {tgt_lang} 字幕本地化与质量保证。

## 2. 核心任务
1. **主题总结 (`theme`)**：用两句话精炼概括当前文本块的核心主题与讨论要点。
2. **权威术语核对与别名提取**：下方 `# terms_note` 是当前文本块中预先匹配出的权威参考术语列表。
   - 检查原文 `<text>` 中是否存在这些权威术语的 **ASR 语音/拼写讹误**（如因发音相近被语音转文字打错的词）。若有，将标准术语填入 `src`，标准译名填入 `tgt`，并将原文中出现的具体 ASR 错词/同音词填入 `aliases` 数组中。
   - **绝对严禁**：严禁将名词单复数（如 shotgun/shotguns）、名词所有格（`'s`）或动词时态词形变化（`-ed`, `-ing`）等正常语法变形当作 ASR 错词填入 `aliases`！`aliases` **仅限真正的 ASR 拼写/同音识别错误**（例如 `"gally"` 对应 `"Gjallarhorn"`，`"trench barrel"` 被 ASR 识别成拼写错词等）。
3. **提取新术语**：提取原文 `<text>` 中未在下方列出但属于专有名词、武器、技能、游戏/业务机制、地名、人名、领域行话的新术语，并给出准确的 {tgt_lang} 译名。
   - 常用通用词汇（如 system, player, video, update 等字面直译无歧义词）严禁提取。
   - `tgt` 必须为干净无括号的中文译名，解释说明放入 `note`。

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
      "aliases": ["原文中出现的 ASR 语音/拼写讹误（非语法变形，无则留空数组 []）"],
      "note": "简要上下文或术语解释"
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

    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[str(i)] = {
            "origin": line,
            "direct": f"<<请逐行翻译为{TARGET_LANGUAGE}，确保忠实于原文，且仅输出翻译后结果>>",
        }
    json_format = json.dumps(json_dict, indent=4, ensure_ascii=False)

    prompt_faithfulness = f"""
## Role: Expert Linguistic Translator
{style_prompt}
## Task Description:
Translate the provided {SRC_LANGUAGE} text lines (from video context) into {TARGET_LANGUAGE} line by line. The primary goal is to **faithfully** reproduce the original meaning, tone, and nuances, while also ensuring the translation **conforms to the expressive conventions of {TARGET_LANGUAGE}**.

## Context:
- Source Language: {SRC_LANGUAGE}
- Target Language: {TARGET_LANGUAGE}
- Content: Video subtitles/transcript lines.
- Terminology & Context: Provided below under 'Terminology & Shared Context'.

## Translation Guidelines:
1. **Line-by-Line Translation**: Translate each line independently while maintaining semantic continuity from adjacent context lines.
2. **Faithful Rendition**: Accurately convey the original semantics, tone (formal, informal, enthusiastic, serious), and subtle nuances.
3. **Natural Flow and Cognitive Alignment**:
   - The translation must align with {TARGET_LANGUAGE}'s patterns of thought, reading habits, and spoken rhythm.
   - Natural clause punctuation: You may naturally add Chinese commas (，) between clauses to improve readability and break up stiff, robotic machine-translation run-on sentences.
   - Disfluency removal: Remove meaningless spoken filler words (`uh`, `um`, `you know`, `like`), producing clean, professional subtitles.
4. **Mandatory Terminology Consistency**:
   - Strictly adhere to the official Chinese translations in `### 术语库-词典`.
   - **Inflection Rule**: When a term appears in the original text in plural form, possessive (`'s`), or verb inflections (`-ed`, `-ing`), you **MUST** use the canonical official translation from `### 术语库-词典` corresponding to its base term, integrating it accurately and grammatically into the sentence. It is strictly forbidden to invent alternative translations or translate it literally.
5. **Proper Noun Handling**:
   - Preserve unlisted proper nouns (people, places, brands, model numbers) in their original form unless a standard recognized {TARGET_LANGUAGE} equivalent exists.
6. **No Additions or Parentheses**:
   - Do NOT add annotations, explanations, or source terms in parentheses (e.g., translate "Still Hunt" to "静谧狩猎", NEVER "静谧狩猎 (Still Hunt)").

## Reference Examples:
* Original: "Hey guys, welcome back to the channel!" -> Translation: "大家好，欢迎回到频道！" (Informal, direct, aligns with spoken language habits, natural punctuation)
* Original: "The algorithm uses a Still Hunt approach." (Assuming "Still Hunt: 静谧狩猎" is in glossary) -> Translation: "该算法使用静谧狩猎方法。" (Strict terminology adherence, natural word order)
* Original: "This is powered by the new Nvidia RTX 4090." -> Translation: "这由新的英伟达 RTX 4090 驱动。" (Proper noun preserved, structure is natural)
* Original: "Crew and Alex, for example, are both on Trench Barrel aggressive shotguns..." (Assuming Crew/Alex not in glossary, "Trench Barrel: 战壕枪管", "aggressive frame: 攻击架构") -> Translation: "例如，Crew和Alex都装备了带有战壕枪管的攻击型霰弹枪..." (Terminology strictly preserved across inflection 'shotguns' -> '霰弹枪', names preserved, natural syntax)

## Input Text Lines:
<source_lines>
{lines}
</source_lines>

## Terminology & Shared Context:
{shared_prompt}

## Output Format:
Provide the translation results strictly in the following JSON format. Fill in the `"direct"` field for each line number with your {TARGET_LANGUAGE} translation.

```json
{json_format}
```

Think step-by-step to ensure all instructions and constraints are met before generating the final JSON output.
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
   - Ensure effortless reading flow and authentic auditory cadence (the natural rhythm of spoken {TARGET_LANGUAGE}).
   - Remove spoken hesitation fillers (`uh`, `um`, `you know`) and awkward translationese or stacked literal calques.
   - Natural clause punctuation: You may naturally add Chinese commas (，) between clauses to improve readability and sentence rhythm.
4. **Mandatory Terminology & Proper Noun Preservation**:
   - **Strict Terminology Enforcement**: Strictly enforce the provided glossary and dictionary in `### 术语库-词典`. Every glossary translation correctly used in 'direct' MUST remain intact in 'free' — never abbreviate, paraphrase, or alter official terms during polishing. When terms appear with inflections (plurals, `'s`, `-ed`, `-ing`), maintain their standard canonical translation.
   - **Proper Nouns**: Ensure proper nouns (not in glossary) remain intact or in their standard {TARGET_LANGUAGE} form.
5. **No Additions / Annotations**:
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
