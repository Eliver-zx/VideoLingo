import json
from core.utils import get_source_language, load_key, load_key_or

## ================================================================
# @ _3_1_split_nlp.py
def get_punctuation_prompt(text: str, language: str | None = None):
    lang = language or get_source_language()
    return f"""
## Role
You are a speech transcript punctuation specialist in **{lang}**.

## Task
Restore sentence-ending punctuation (`.`, `?`, `!` or `。`, `？`, `！`) and internal commas into the given ASR transcript segment so that it can be split into complete, natural conversational sentences.

## Rules
1. **Strict Word Preservation**: NEVER change, add, delete, or reorder any word, character, or number from the input text. Only insert or fix punctuation marks.
2. **Complete Natural Sentences**: Place a sentence-ending mark (`.`, `?`, `!`, `。`, `？`, `！`) at the end of every complete conversational sentence (one full thought).
3. **Break Spoken Run-ons & Comma Splices**: In spoken transcripts, speakers often chain multiple independent clauses together using commas or filler conjunctions (`and`, `so`, `but`, `then`, `because`, `然后`, `所以`, `但是`, `而且`). Whenever the preceding clause already expresses a complete thought, end it with a period (`.` or `。`) instead of chaining a long run-on sentence with commas.
4. **No Fragmentation**: Keep dependent clauses, short modifiers, and prepositional phrases attached to their main clause; do not put periods inside a single incomplete phrase.
5. **Acoustic Pause Markers (`‖`)**: The symbol `‖` in the input indicates a spoken acoustic pause (>= 0.4s). Use `‖` as a cue for placing a period or comma at natural clause boundaries, and **do not include `‖` in the output**.

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
# @ step4_splitbymeaning.py
def get_split_prompt(sentence, num_parts = 2, word_limit = 20, language = None):
    language = language or get_source_language()
    split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
Split the given subtitle text into **{num_parts}** parts, each less than **{word_limit}** words.

1. Maintain sentence meaning coherence according to Netflix subtitle standards
2. MOST IMPORTANT: Keep parts roughly equal in length (minimum 3 words each)
3. Split at natural points like punctuation marks or conjunctions
4. If provided text is repeated words, simply split at the middle of the repeated words.

## Steps
1. Analyze the sentence structure, complexity, and key splitting challenges
2. Generate two alternative splitting approaches with [br] tags at split positions
3. Compare both approaches highlighting their strengths and weaknesses
4. Choose the best splitting approach

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief description of sentence structure, complexity, and key splitting challenges",
    "split1": "First splitting approach with [br] tags at split positions",
    "split2": "Alternative splitting approach with [br] tags at split positions",
    "assess": "Comparison of both approaches highlighting their strengths and weaknesses",
    "choice": "1 or 2"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
    return split_prompt

## ================================================================
# @ step4_1_summarize.py
def get_summary_prompt(source_content, custom_terms_json=None):
    src_lang = get_source_language()
    tgt_lang = load_key("target_language")
    
    # add custom terms note
    terms_note = ""
    if custom_terms_json and custom_terms_json.get('terms'):
        terms_list = []
        for term in custom_terms_json['terms']:
            terms_list.append(f"- {term['src']}: {term['tgt']} ({term.get('note', '')})")
        terms_note = (
            "\n### Reference & Existing Terms\n"
            "Exclude exact matches already listed below, but DO extract any ASR spelling/phonetic variants "
            "of these terms that appear in the text and map them to the same translation:\n"
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
   - **Proper Nouns & Entities**: Names of people, brands, studios, products, games, locations, specific weapons/skills/mechanisms.
   - **Domain Jargon & Polysemous Terms**: Words that have an everyday meaning but carry a specialized meaning in this video's context (where literal translation would be wrong).
   - **ASR Misrecognitions / Spelling Variants**: Terms in `<text>` that are phonetically misrecognized or misspelled by ASR. Extract the **exact surface form as written in `<text>`** in `src`, and provide the correct `{tgt_lang}` translation in `tgt`.
   - **Technical Acronyms & Code Names**.
2. **Counterfactual Exclusion Test (Strict Filter)**:
   - Before extracting any term, ask: *"If this word is NOT in the glossary and a translator translates it literally, will it be wrong or inconsistent?"*
   - If a literal translation is already unambiguous and accurate (e.g., generic words like `system`, `video`, `performance`, `database`, `algorithm`, `player`, `website`, `update`), **DO NOT extract it**.
3. **Translation Requirement**:
   - Provide the accurate `{tgt_lang}` translation in `tgt` (only universally kept acronyms like `AI`, `API`, `CPU`, `GPU` may stay in their original form).

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
      "src": "exact {src_lang} term or ASR variant in text",
      "tgt": "{tgt_lang} translation", 
      "note": "Brief context or note (mention standard spelling if ASR variant)"
    }}
  ]
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return summary_prompt

## ================================================================
# @ step5_translate.py & translate_lines.py
def generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt):
    prev_text = "\n".join(previous_content_prompt) if isinstance(previous_content_prompt, list) else (previous_content_prompt or "")
    after_text = "\n".join(after_content_prompt) if isinstance(after_content_prompt, list) else (after_content_prompt or "")
    return f'''### Context Information
<previous_content>
{prev_text}
</previous_content>

<subsequent_content>
{after_text}
</subsequent_content>

### Content Summary
{summary_prompt or ""}

### Points to Note
{things_to_note_prompt or ""}'''

def get_translation_style_prompt():
    """The style the user asks for, as a section of the translation prompt; empty without one."""
    style = str(load_key_or("translation_style", "") or "").strip()
    if not style:
        return ""
    return f'''### Translation Style
The user asks for the following style. Follow it in the translation, without changing the meaning of the original:
<translation_style>
{style}
</translation_style>

'''

def get_prompt_faithfulness(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    # The polish step applies the style when there is one, this step stays a direct translation
    style_prompt = "" if load_key("reflect_translate") else get_translation_style_prompt()
    # Split lines by \n
    line_splits = lines.split('\n')
    
    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[f"{i}"] = {"origin": line, "direct": f"direct {TARGET_LANGUAGE} translation {i}."}
    json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)

    src_language = get_source_language()
    prompt_faithfulness = f'''
## Role
You are a professional Netflix subtitle translator, fluent in both {src_language} and {TARGET_LANGUAGE}, as well as their respective cultures. 
Your expertise lies in accurately understanding the semantics and structure of the original {src_language} text and faithfully translating it into {TARGET_LANGUAGE} while preserving the original meaning.

## Task
We have a segment of original {src_language} subtitles that need to be directly translated into {TARGET_LANGUAGE}. These subtitles come from a specific context and may contain specific themes and terminology.

1. Translate the original {src_language} subtitles into {TARGET_LANGUAGE} line by line
2. Ensure the translation is faithful to the original, accurately conveying the original meaning
3. Consider the context and professional terminology

{shared_prompt}

{style_prompt}<translation_principles>
1. **Faithful & Context-Resolved**: Accurately convey the content and meaning of the original text without arbitrarily adding or omitting information. Use `<previous_content>`, `<subsequent_content>`, and `Content Summary` to resolve ambiguous pronouns, omitted subjects, and polysemous words.
2. **ASR Homophone & Phonetic Error Recovery**: The source subtitles are generated by Automatic Speech Recognition (ASR) and may contain phonetic misrecognitions, homophone errors, or misspellings of technical terms and proper nouns. Infer and translate the speaker's intended meaning from context rather than translating literal transcription errors.
3. **Strict Terminology & Clean Proper Nouns**: Strictly follow the translations in `Points to Note`. For person names, model numbers, version codes, or code identifiers not listed in the glossary, keep their original form. **Never** append parenthetical source text, original terms in brackets, or translator notes inside the subtitle lines.
4. **Line-Locked Alignment**: Maintain strict 1-to-1 line alignment with the numbered input lines. Connect clauses smoothly across adjacent lines, but never move semantic content from one line into another, as each line is locked to a specific video timestamp.
5. **Natural Clause Punctuation**: Preserve natural internal clause commas and pauses within each line so downstream subtitle splitting can segment long lines cleanly.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_faithfulness.strip()


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    json_format = {
        key: {
            "origin": value.get("origin", ""),
            "direct": value["direct"],
            "reflect": "your reflection on direct translation",
            "free": "your free translation"
        }
        for key, value in faithfulness_result.items()
    }
    json_format = json.dumps(json_format, indent=2, ensure_ascii=False)

    src_language = get_source_language()
    style_prompt = get_translation_style_prompt()
    prompt_expressiveness = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {TARGET_LANGUAGE} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles.
Your task is to reflect on and improve these direct translations to create more natural and fluent {TARGET_LANGUAGE} subtitles.

1. Analyze the direct translation results line by line, pointing out existing issues
2. Provide detailed modification suggestions
3. Perform free translation based on your analysis
4. Do not add comments, parenthetical source terms, or explanations in the translation, as the subtitles are for the audience to read
5. Do not leave empty lines in the free translation, as the subtitles are for the audience to read

{shared_prompt}

{style_prompt}<Translation Analysis Steps>
Please use a two-step thinking process to handle the text line by line:

1. Direct Translation Reflection (`reflect` — 4-Point SubMQM Diagnostic):
   - **Accuracy & ASR Recovery**: Check against `origin` for mistranslations, unrecovered ASR homophone/phonetic errors, or shifted meaning across line boundaries.
   - **Terminology & Cleanliness**: Verify exact adherence to `Points to Note` and ensure no parenthetical source terms or translator notes were inserted.
   - **Anti-Translationese**: Identify awkward literal calques, stacked modifiers, unnatural passive constructions, or rigid source-language word order that sounds robotic in {TARGET_LANGUAGE}.
   - **Spoken Cadence & Conciseness**: Check whether the line reads like natural spoken {TARGET_LANGUAGE} within a concise subtitle length while preserving natural internal clause commas.

2. {TARGET_LANGUAGE} Free Translation (`free` — Targeted Refinement & Keep-if-Flawless):
   - **Principle of Necessary Enhancement (Keep-if-Flawless)**: If `direct` already passes all four diagnostic checks and sounds like natural native {TARGET_LANGUAGE} speech, write `"OK"` in `reflect` and copy `direct` **unchanged** into `free`. Do not rewrite or substitute synonyms just for the sake of making changes.
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

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_expressiveness.strip()


## ================================================================
# @ step6_splitforsub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    targ_lang = load_key("target_language")
    src_lang = get_source_language()
    src_splits = src_part.split('\n')
    num_parts = len(src_splits)
    src_part = src_part.replace('\n', ' [br] ')
    align_parts_json = ','.join(
        f'''
        {{
            "src_part_{i+1}": "{src_splits[i]}",
            "target_part_{i+1}": "Corresponding aligned {targ_lang} subtitle part"
        }}''' for i in range(num_parts)
    )

    align_prompt = f'''
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
{src_lang} Original: "{src_sub}"
{targ_lang} Original: "{tr_sub}"
Pre-processed {src_lang} Subtitles ([br] indicates split points): {src_part}
</subtitles>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of word order, structure, and semantic correspondence between two subtitles",
    "align": [
        {align_parts_json}
    ]
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return align_prompt

## ================================================================
# @ step8_gen_audio_task.py @ step10_gen_audio.py
def get_subtitle_trim_prompt(text, duration):
 
    rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    - "Let's discuss the various different perspectives on this topic" can be shortened to "Let's discuss different perspectives on this topic"
    - "Can you describe in detail your experience from yesterday" can be shortened to "Can you describe yesterday's experience" '''

    trim_prompt = f'''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors. 
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
Subtitle: "{text}"
Duration: {duration} seconds
</subtitles>

## Processing Rules
{rule}

## Processing Steps
Please follow these steps and provide the results in the JSON output:
1. Analysis: Briefly analyze the subtitle's structure, key information, and filler words that can be omitted.
2. Trimming: Based on the rules and analysis, optimize the subtitle by making it more concise according to the processing rules.

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of the subtitle, including structure, key information, and potential processing locations",
    "result": "Optimized and shortened subtitle in the original subtitle language"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return trim_prompt

## ================================================================
# @ tts_main
def get_correct_text_prompt(text):
    return f'''
## Role
You are a text cleaning expert for TTS (Text-to-Speech) systems.

## Task
Clean the given text by:
1. Keep only basic punctuation (.,?!)
2. Preserve the original meaning

## INPUT
{text}

## Output in only JSON format and no other text
```json
{{
    "text": "cleaned text here"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
