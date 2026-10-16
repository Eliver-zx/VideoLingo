import inspect
import re
import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._1_ytdlp import find_subtitle_file
from core._3_2_split_meaning import split_sentence as _default_split_sentence
from core.task_runner import StopTask
from core.prompts import get_align_prompt as _default_get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.config_utils import get_text_joiner
from core.utils.models import *
console = Console()

split_sentence = _default_split_sentence
get_align_prompt = _default_get_align_prompt

_LEADING_PUNCT_RE = re.compile(r'^[\s，。、“”‘’！!？?：:；;（）()【】\[\]《》<>…—\-·]+')
_CLAUSE_PUNCT = set(',;:，、；：—')
_BREAK_CONJUNCTIONS = {
    'and', 'but', 'or', 'so', 'because', 'since', 'although', 'though',
    'while', 'when', 'where', 'which', 'that', 'who', 'if', 'unless',
    'until', 'before', 'after', 'whether', 'as', 'yet', 'nor',
    'to', 'for', 'with', 'in', 'on', 'at', 'from', 'by', 'about',
}

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def get_target_parts(response_data, num_parts):
    """The aligned target parts of a response, or None unless there are exactly `num_parts` non-empty ones."""
    align_data = response_data.get('align') if isinstance(response_data, dict) else None
    if not isinstance(align_data, list) or len(align_data) != num_parts:
        return None
    parts = []
    for i, item in enumerate(align_data):
        part = item.get(f'target_part_{i+1}') if isinstance(item, dict) else None
        if not isinstance(part, str) or not part.strip():
            return None
        parts.append(part.strip())
    return parts

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split('\n')
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if get_target_parts(response_data, len(src_parts)) is None:
            return {"status": "error", "message": f"Align does not contain {len(src_parts)} non-empty parts as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    tr_parts = get_target_parts(parsed, len(src_parts))
    
    # The parts are in the target language, so the source-language joiner does not apply
    tr_remerged = join_words(tr_parts, get_text_joiner(''.join(tr_parts)))
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def _find_spaced_cut(words: List[str], weights: List[float], start_idx: int, end_idx: int, target_ratio: float) -> int:
    """Find a phrase-aware word cut index in (start_idx, end_idx) near target_ratio."""
    if end_idx - start_idx <= 1:
        return start_idx + 1
    seg_weights = weights[start_idx:end_idx]
    total_w = sum(seg_weights) or float(end_idx - start_idx)
    target_w = target_ratio * total_w

    cum = 0.0
    ideal_cut = start_idx + 1
    min_diff = float('inf')
    for idx in range(start_idx + 1, end_idx):
        cum += weights[idx - 1]
        diff = abs(cum - target_w)
        if diff < min_diff:
            min_diff = diff
            ideal_cut = idx

    window = 3
    low = max(start_idx + 1, ideal_cut - window)
    high = min(end_idx - 1, ideal_cut + window)

    best_cut = ideal_cut
    best_score = -float('inf')
    for cut in range(low, high + 1):
        prev_word = words[cut - 1]
        next_word = words[cut]
        if prev_word.endswith('-') or next_word.startswith('-'):
            continue
        dist_penalty = abs(cut - ideal_cut) * 0.35
        bonus = 0.0
        if prev_word and prev_word[-1] in _CLAUSE_PUNCT:
            bonus = 2.0
        else:
            clean_next = re.sub(r'^[^\w]+|[^\w]+$', '', next_word).lower()
            if clean_next in _BREAK_CONJUNCTIONS:
                bonus = 1.1
        score = bonus - dist_penalty
        if score > best_score:
            best_score = score
            best_cut = cut
    return best_cut

def align_source_by_ratio(src_sub: str, tr_sub: str, tr_split: str) -> Tuple[List[str], List[str], str]:
    """Split source sentence locally according to target-language split parts and phrase boundaries."""
    tr_parts = [p.strip() for p in tr_split.split('\n') if p.strip()]
    tr_remerged = join_words(tr_parts, get_text_joiner(''.join(tr_parts))) if tr_parts else str(tr_sub)
    if len(tr_parts) < 2:
        return [src_sub], [tr_sub], str(tr_sub)

    tr_lens = [max(1.0, calc_len(p)) for p in tr_parts]
    total_tr_len = sum(tr_lens)
    ratios = [length / total_tr_len for length in tr_lens]
    num_parts = len(tr_parts)

    src_joiner = get_text_joiner(src_sub)
    if src_joiner == " ":
        src_words = src_sub.split()
        if len(src_words) < num_parts:
            return [src_sub], [tr_sub], tr_remerged
        weights = [max(1.0, calc_len(w)) for w in src_words]
        src_parts = []
        start_idx = 0
        rem_ratio = 1.0
        for part_idx in range(num_parts - 1):
            parts_left = num_parts - part_idx
            max_end = len(src_words) - (parts_left - 1)
            rel_ratio = ratios[part_idx] / rem_ratio if rem_ratio > 0 else 1.0 / parts_left
            cut_idx = _find_spaced_cut(src_words, weights, start_idx, len(src_words), rel_ratio)
            cut_idx = min(max(start_idx + 1, cut_idx), max_end)
            src_parts.append(" ".join(src_words[start_idx:cut_idx]))
            start_idx = cut_idx
            rem_ratio = max(1e-6, rem_ratio - ratios[part_idx])
        src_parts.append(" ".join(src_words[start_idx:]))
    else:
        src_chars = list(src_sub.strip())
        if len(src_chars) < num_parts:
            return [src_sub], [tr_sub], tr_remerged
        weights = [max(1.0, calc_len(c)) for c in src_chars]
        src_parts = []
        start_idx = 0
        rem_ratio = 1.0
        for part_idx in range(num_parts - 1):
            parts_left = num_parts - part_idx
            max_end = len(src_chars) - (parts_left - 1)
            rel_ratio = ratios[part_idx] / rem_ratio if rem_ratio > 0 else 1.0 / parts_left
            seg_w = sum(weights[start_idx:])
            target_w = rel_ratio * seg_w
            cum = 0.0
            ideal_cut = start_idx + 1
            min_diff = float('inf')
            for idx in range(start_idx + 1, max_end + 1):
                cum += weights[idx - 1]
                diff = abs(cum - target_w)
                if diff < min_diff:
                    min_diff = diff
                    ideal_cut = idx
            window = 4
            low = max(start_idx + 1, ideal_cut - window)
            high = min(max_end, ideal_cut + window)
            punct_cuts = [i for i in range(low, high + 1) if src_chars[i - 1] in _CLAUSE_PUNCT]
            cut_idx = min(punct_cuts, key=lambda i: abs(i - ideal_cut)) if punct_cuts else ideal_cut
            src_parts.append("".join(src_chars[start_idx:cut_idx]).strip())
            start_idx = cut_idx
            rem_ratio = max(1e-6, rem_ratio - ratios[part_idx])
        src_parts.append("".join(src_chars[start_idx:]).strip())

    if any(not p.strip() for p in src_parts) or len(src_parts) != len(tr_parts):
        return [src_sub], [tr_sub], tr_remerged

    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)

    return src_parts, tr_parts, tr_remerged

def _call_split_sentence(text: str, num_parts: int = 2, language: str = None) -> str:
    if language is not None:
        try:
            sig = inspect.signature(split_sentence)
            if "language" in sig.parameters:
                return split_sentence(text, num_parts=num_parts, language=language)
        except (TypeError, ValueError):
            pass
    return split_sentence(text, num_parts=num_parts)

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    target_language = load_key_or("target_language", None)
    use_legacy_align = (get_align_prompt is not _default_get_align_prompt)
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        if use_legacy_align:
            split_src = _call_split_sentence(src_lines[i], num_parts=2).strip()
            if len([part for part in split_src.split('\n') if part.strip()]) < 2:
                console.print(f"[yellow]⚠️ Line {i} could not be split, keeping it as is[/yellow]")
                return
            src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        else:
            tr_clean = _LEADING_PUNCT_RE.sub('', str(tr_lines[i])).strip() or str(tr_lines[i]).strip()
            split_tr = _call_split_sentence(tr_clean, num_parts=2, language=target_language).strip()
            if len([part for part in split_tr.split('\n') if part.strip()]) < 2:
                console.print(f"[yellow]⚠️ Line {i} could not be split, keeping it as is[/yellow]")
                return
            src_parts, tr_parts, tr_remerged = align_source_by_ratio(str(src_lines[i]), str(tr_lines[i]), split_tr)
        # Source and translation must stay in step: a line is split on both sides or not at all
        if not tr_parts or len(src_parts) < 2 or len(src_parts) != len(tr_parts):
            console.print(f"[yellow]⚠️ Line {i} was aligned into a different number of parts, keeping it as is[/yellow]")
            return
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        futures = [executor.submit(process, i) for i in to_split]
    for i, future in zip(to_split, futures):
        error = future.exception()
        if isinstance(error, StopTask):
            raise error
        if error:
            console.print(f"[yellow]⚠️ Line {i} is kept unsplit: {error}[/yellow]")
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines, remerged_tr_lines

def split_source_lines(lines: List[str]) -> List[str]:
    """Split the lines that are too long for a subtitle, for subtitles without a translation."""
    max_length = load_key("subtitle")["max_length"]

    def process(line):
        if len(line) <= max_length:
            return [line]
        parts = [part.strip() for part in split_sentence(line, num_parts=2).split('\n') if part.strip()]
        return parts if len(parts) > 1 else [line]

    for _ in range(3):
        if all(len(line) <= max_length for line in lines):
            break
        check_cancel()
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            lines = [part for parts in executor.map(process, lines) for part in parts]
    return lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()

    if find_subtitle_file():
        # The subtitles of the user keep their lines and their times
        console.print("[cyan]📄 Subtitles of the user: the lines are kept as they are[/cyan]")
        for path in (_5_SPLIT_SUB, _5_REMERGED):
            pd.DataFrame({'Source': src, 'Translation': trans}).to_excel(path, index=False)
        return
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
