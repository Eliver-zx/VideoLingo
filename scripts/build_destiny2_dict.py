#!/usr/bin/env python3
"""Build and clean the authoritative Destiny 2 dictionary (`glossary/destiny2/dict.json`) from `old/1/term-map.json`.

Resolves 56 homonyms using domain-aware category priorities and canonical Destiny 2 resolutions,
and purges conversational emotes, quest-step phrases, and generic everyday UI single words.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

TERM_MAP_PATH = Path("old/1/term-map.json")
OUTPUT_DICT_PATH = Path("glossary/destiny2/dict.json")
TERMS_CSV_PATH = Path("glossary/destiny2/terms.csv")
PERSISTED_LORE_PATH = Path("glossary/destiny2/lore_terms.json")
DOWNLOADS_BOOKS_PATH = Path("/Users/eliver/Downloads/lore-books.json")
DOWNLOADS_LORE_PATH = Path("/Users/eliver/Downloads/lore.json")
ROMAN_NUMERAL_PATTERN = re.compile(r"^(?:[ivxlcdm]+|[0-9]+)[\s.:–-]*$", re.IGNORECASE)
CHAPTER_PREFIX_PATTERN = re.compile(
    r"^(?:(?:[IVXLCDM]{1,6}|[0-9]{1,3})\s*[:.\-–—|]\s+|Entry\s+[0-9IVXLCDM]+|Chapter\s+[0-9IVXLCDM]+|Part\s+[0-9IVXLCDM]+)",
    re.IGNORECASE,
)
CONVERSATIONAL_PREFIX_PATTERN = re.compile(
    r"^(?:What If\b|How to\b|Why\b|Where\b|When\b|Who\b|Could\b|Would\b|Should\b|Can we\b|If we\b|Do you\b|Did you\b|Have you\b|Pleased to\b|You get\b)",
    re.IGNORECASE,
)

# Category priority (earlier categories take priority over later ones)
CATEGORY_PRIORITY: list[str] = [
    "classes",
    "damage_types",
    "perks",
    "modifiers",
    "activity_types",
    "activity_modes",
    "destinations",
    "activities",
    "categories",
    "stats",
    "custom",
    "items",
]

# Canonical resolutions for homonyms and domain-standard terms
CANONICAL_HOMONYM_RESOLUTIONS: dict[str, str] = {
    "Titan": "泰坦",
    "Kinetic": "动能",
    "Solar": "烈日",
    "Arc": "电弧",
    "Void": "虚空",
    "Sparrows": "快雀",
    "Sparrow": "快雀",
    "The Last City": "圣城",
    "Vanguard Ops": "先锋行动",
    "Tomorrow's Answer": "明日答案",
    "Tomorrow's Answer (Adept)": "明日答案（专家）",
    "Surrounded": "腹背受敌",
    "Together Forever": "长相聚",
    "Worm God Incarnation": "虫神化身",
    "Pulse Rifle Scavenger": "脉冲步枪弹药回收器",
    "Unstoppable Hand Cannon": "势不可挡手炮",
    "Overload Hand Cannon": "超载手炮",
    "Unstoppable Grenade Launcher": "势不可挡榴弹发射器",
    "Snapload Finisher": "速装终结技",
    "Guiding Light": "指引之光",
    "Kinetic Dexterity": "动能武器灵巧",
    "Unflinching Kinetic Aim": "稳定动能武器瞄准",
    "Match Game": "元素匹配",
    "Raid": "突袭",
    "Salvage": "回收",
    "Shock and Awe": "震慑行动",
    "Lightning Strikes Twice": "两次闪电打击",
    "Radiant Largesse": "光辉慷慨",
    "Energized": "充能",
    "Reflection Effects": "映像效果",
    "Heartbreaker": "碎心者",
    "Praetorian Ornament": "禁卫军皮肤",
    "Event Pack": "活动包",
    "Rapid Response": "快速反应",
    "Momentum": "动量",
    "Heat of the Moment": "瞬时热情",
    "Shocker": "震荡",
    "Clash": "死斗",
    "Control": "占领",
    "Elimination": "灭绝",
    "Mayhem": "鏖战",
    "Rumble": "混战",
    "Supremacy": "霸权",
    "Survival": "生存",
    "Rift": "裂痕",
    "Doubles": "双打",
    "Nightfall: The Pyramidion": "日落：金字塔",
    "Nightfall: The Inverted Spire": "日落：倒置尖塔",
    "Nightfall: Exodus Crash": "日落：移民号的坠毁",
    "Nightfall: The Arms Dealer": "日落：军火交易商",
    "Nightfall: Savathûn's Song": "日落：萨瓦图恩之歌",
    "Iron Banner Clash": "铁旗死斗",
    "Iron Banner Control": "铁旗占领",
    "Iron Banner Supremacy": "铁旗霸权",
    "Nightfall: A Garden World": "日落：锦绣世界",
    "Nightfall: Tree of Probabilities": "日落：概率之树",
    "Emblem": "徽标",
    "Guardian": "守护者",
    "Triumph": "成就",
}

# Conversational emotes, quest-step phrases, and generic everyday single words to purge
PURGE_TERMS: set[str] = {
    # Conversational emote / quest phrases
    "You're Welcome",
    "And Another Thing",
    "Once More",
    "Out of Place",
    "At Any Cost",
    "The Right Choice",
    "Over and Done With",
    "A Bad Run",
    "Stronger Than Now",
    "Here I Stand",
    "Where I Belong",
    "Mind of Its Own",
    "The Thick of It",
    "In With the New",
    "Yes, Queen",
    "The Past",
    "Past Is Past",
    "New Tricks",
    "Cleans Up Good",
    "Unequip All",
    "No Projection",
    "A Challenge Awaits…",
    "Quickplay: Normal",
    "Quickplay: Normal".replace(":", "："),
    "Wait For It…",
    "From Zero…",
    "…To Hero",
    "This… Is… TRIALS!",
    "You Think You're Big Time?",
    "You Think Distance Is Your Ally?",
    "Thar Be Treasure!",
    "Wake Up!",
    "Ding!",
    "Boom!",
    "Like a Moth to a…",
    "Ain't My First Rodeo",
    "I Am the Boss Now",
    "Go About Your Business",
    "Never Live It Down",
    "Ahead of Its Time",
    "Charmed, I'm Sure",
    "Don't Look Back",
    "At Least It's a Cape",
    "It's Cool",
    "Keep It Clean",
    "Salute to the Colonel",
    "A Single Clap",
    "A Little Privacy, Please",
    "The Iron Banner Has Returned!",
    "Solstice is here!",
    "Alas, It Happened Emote Pack",
    "Pisces What You Did There Emote Pack",
    "Best Friend Emote Pack",
    "Fireteam Fool Emotes",
    "Interstellar Warfare Emotes Pack",
    "A Hat Trick of Emotes Pack",
    "Frontier Emote Pack",
    "Trials of Osiris Emote Pack",
    "Season of the Risen Exotic Emote Pack",
    "Season of the Haunted Exotic Emote Pack",
    "Season of Plunder Exotic Emote Pack",
    "This Ghost has been upgraded to maximum capacity and has access to its additional mod slot.",
    "RENEGADE ABILITY SELECTION INSTRUCTIONS",
    "The Game is Afoot!",
    "Arach-NO!",
    "Enhance!",
    "No Ammo? No Problem",
    "You Gonna Eat That Pack",
    "What If",
    "They Had Built",
    "By My Blade",
    "Up My Sleeve",
    "Off My Lawn",
    "On Your Sleeve",
    "You Were Already Dead",
    "I Am Alive",
    "Filamentary, My Dear Ghost",
    "I Said What I Fragment",
    "From Whence They Came",
    "They All Fall Down",
    "I Want Him Dead",
    "My Captain",
    "Your Light in Dark Times",
    "Fight! Projection",
    "Collect! Projection",
    "Bank! Projection",
    "Invade! Projection",
    "Attack! Projection",
    # Quest objectives and patrol phrases
    "Defeat enemies",
    "Defeat Enemies",
    "Defeat the Target",
    "Collect Hive Tablets",
    "Collect Supplies",
    "Collect Candy",
    "Talk to the Drifter",
    "Open the Gate",
    "Destroy the Brain",
    "Return to the Arbor of Light",
    "Stop and Go",
    "Survey mission",
    "Survey the Area",
    "Survey the Hive",
    "Combat Mission",
    "Combat mission",
    "Salvage Mission",
    "Salvage mission",
    "Fragment of Graviton Spike Catalyst acquired",
    "Attune to The When and Where",
    "A New Season Arrives at Eververse",
    "Welcome to the Hall of Heroes",
    "UNLOCKED BY QUEST",
    "Locked Armor Mod",
    "Locked Artifice Socket",
    # Kepler mechanics and instructions
    "Shoot rotatable relays to change their alignment, altering the path of Relocator Launcher shots",
    "Access Matterspark shortcuts across Kepler.",
    "Deal increased Matterspark damage.",
    "Health regeneration begins immediately upon activating Matterspark.",
    "Auto-reload all weapons shortly after entering Matterspark.",
    "Relocator teleportation applies a Void overshield.",
    "Effect lasts for 30 minutes.",
    # Generic everyday single English words (UI labels, common nouns/verbs/foods)
    "Help",
    "All",
    "General",
    "Normal",
    "Soon",
    "Information",
    "Messages",
    "Social",
    "Story",
    "Explore",
    "Career",
    "Hardware",
    "Secrets",
    "Confidence",
    "Doubt",
    "Purpose",
    "Anticipation",
    "Forgiveness",
    "Farewell",
    "Togetherness",
    "Packages",
    "Accessories",
    "Materials",
    "Consumables",
    "Currencies",
    "Playlists",
    "Collections",
    "Vendor",
    "Vehicle",
    "Watermelon",
    "Mint",
    "Cocoa",
    "Candy",
    "Ingredient",
    "Pantry",
    "Butterfly",
    "Six",
    "Attack",
    "Defense",
    "Speed",
    "Boost",
    "Zoom",
    "Release",
    "Wait",
    "Stop",
    "Look",
    "Laugh",
    "Cry",
    "Wave",
    "Dance",
    "Cheer",
    "Point",
    "Salute",
    "Clap",
    "Shrug",
    "Nod",
    "Thanks",
    "Sorry",
    "Please",
    "Hello",
    "Caution",
    "Movement",
}


def is_noise_entry(en: str, purge_lower: set[str]) -> bool:
    """Check if an English term is noise, a tutorial sentence, a quest instruction, or generic conversational phrase."""
    enl = en.strip().lower()
    if not enl or enl in purge_lower:
        return True
    # Dummy non-alphanumeric entries (e.g. ???, ???????)
    if not re.search(r"[a-zA-Z0-9]", en):
        return True
    # Bungie UI raw substitution placeholders
    if "###" in en or "DestinyNamedSubstitutions" in en:
        return True
    # Control button instructions like [Aim Down Sights] : ... or [Shoot] ...
    if re.match(r"^\[(?:Shoot|Super|Aim Down Sights|Melee)\]", en):
        return True
    # Unlock / locked stubs
    if (
        enl.startswith("unlock:")
        or enl.startswith("unlock ")
        or enl == "unlocked by quest"
    ):
        return True
    # Pure numbers or Roman numerals (e.g. I, II, III, 1, 2)
    if ROMAN_NUMERAL_PATTERN.match(enl):
        return True
    # Roman numeral / number chapter or entry prefixes (exempting multi-capital acronyms like D.F.A., D.A.R.C.I.)
    if not re.search(r"^[A-Z]\.[A-Z]\.", en):
        if CHAPTER_PREFIX_PATTERN.match(en):
            return True
        if CONVERSATIONAL_PREFIX_PATTERN.match(en):
            return True
    # Ellipsis titles (e.g. What It Means…)
    if "…" in en or "..." in en:
        return True
    # Tutorial / mechanics full sentences ending in period (keeping weapon/strike acronyms like D.F.A., D.A.R.C.I., S.A.B.E.R.)
    if (
        en.endswith(".")
        and len(en.split()) >= 3
        and not re.search(r"[A-Z]\.[A-Z]\.?$", en)
    ):
        return True
    # Sentence-length phrases
    if len(en.split()) > 8:
        return True
    # Conversational questions or exclamations
    if (en.endswith("?") or en.endswith("!")) and len(en.split()) >= 2:
        return True
    return False


def extract_lore_terms(purge_lower: set[str]) -> dict[str, str]:
    """Extract non-noisy Lore Book titles, Lore Entry titles, Speaker Attributions, and Lore items.

    If Bungie official datasets exist in /Users/eliver/Downloads, extracts directly from them and updates
    glossary/destiny2/lore_terms.json. Otherwise loads from glossary/destiny2/lore_terms.json.
    """
    lore_terms: dict[str, str] = {}

    def is_invalid_speaker(spk_en: str) -> bool:
        low = spk_en.lower()
        if " to " in low:
            return True
        if any(
            low.startswith(p)
            for p in [
                "author",
                "unknown",
                "overheard",
                "traditional",
                "diary of",
                "excerpt from",
                "research notes",
                "notes of",
                "self-authored",
                "transcription",
                "recorded",
                "recovered",
            ]
        ):
            return True
        return False

    if DOWNLOADS_BOOKS_PATH.exists() and DOWNLOADS_LORE_PATH.exists():
        en_dash_pattern = re.compile(
            r"(?:—|--|–)\s*([A-Za-z0-9][A-Za-z0-9\s.,'’–-]{1,60})$"
        )
        zh_dash_pattern = re.compile(
            r"(?:——|—|--|–)\s*([\u4e00-\u9fa5A-Za-z0-9·•]{1,30})$"
        )

        with open(DOWNLOADS_BOOKS_PATH, "r", encoding="utf-8") as f:
            books_data = json.load(f).get("data", {})
        with open(DOWNLOADS_LORE_PATH, "r", encoding="utf-8") as f:
            lore_data = json.load(f).get("data", {})

        # 1. Lore Book titles
        for book_item in books_data.values():
            name = (book_item.get("name") or "").strip()
            zh = (book_item.get("name_zh") or "").strip()
            if name and zh and not is_noise_entry(name, purge_lower):
                lore_terms[name] = zh

        # 2. Lore Book Entry titles
        for book_item in books_data.values():
            for e in book_item.get("entries", []):
                name = (e.get("name") or "").strip()
                zh = (e.get("name_zh") or "").strip()
                if name and zh and not is_noise_entry(name, purge_lower):
                    lore_terms[name] = zh

        # 3. Speaker attributions from subtitles
        for lore_item in lore_data.values():
            sub_en = (lore_item.get("subtitle") or "").strip()
            sub_zh = (lore_item.get("subtitle_zh") or "").strip()
            if sub_en and sub_zh:
                m_en = en_dash_pattern.search(sub_en)
                m_zh = zh_dash_pattern.search(sub_zh)
                if m_en and m_zh:
                    spk_en = re.sub(r"\.+$", "", m_en.group(1).strip()).strip()
                    spk_zh = m_zh.group(1).strip()
                    if (
                        not is_noise_entry(spk_en, purge_lower)
                        and not is_invalid_speaker(spk_en)
                        and len(spk_en) <= 40
                        and len(spk_zh) <= 20
                    ):
                        lore_terms[spk_en] = spk_zh

        # 4. Speaker attributions from text endings
        def extract_text_speakers(ten: str, tzh: str) -> None:
            lines_en = [line.strip() for line in ten.split("\n") if line.strip()]
            lines_zh = [line.strip() for line in tzh.split("\n") if line.strip()]
            if lines_en and lines_zh:
                last_en = lines_en[-1]
                last_zh = lines_zh[-1]
                m_en = en_dash_pattern.match(last_en) or en_dash_pattern.search(last_en)
                m_zh = zh_dash_pattern.match(last_zh) or zh_dash_pattern.search(last_zh)
                if m_en and m_zh:
                    spk_en = re.sub(r"\.+$", "", m_en.group(1).strip()).strip()
                    spk_zh = m_zh.group(1).strip()
                    if (
                        not is_noise_entry(spk_en, purge_lower)
                        and not is_invalid_speaker(spk_en)
                        and len(spk_en) <= 40
                        and len(spk_zh) <= 20
                    ):
                        lore_terms[spk_en] = spk_zh

        for lore_item in lore_data.values():
            extract_text_speakers(
                lore_item.get("text", "") or "", lore_item.get("text_zh", "") or ""
            )
        for book_item in books_data.values():
            for e in book_item.get("entries", []):
                extract_text_speakers(
                    e.get("text", "") or "", e.get("text_zh", "") or ""
                )

        # 5. Lore Entry names
        for lore_item in lore_data.values():
            name = (lore_item.get("name") or "").strip()
            zh = (lore_item.get("name_zh") or "").strip()
            if name and zh and not is_noise_entry(name, purge_lower):
                lore_terms[name] = zh

        # Persist extracted lore terms
        PERSISTED_LORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PERSISTED_LORE_PATH.write_text(
            json.dumps(lore_terms, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif PERSISTED_LORE_PATH.exists():
        with open(PERSISTED_LORE_PATH, "r", encoding="utf-8") as f:
            lore_terms = json.load(f)

    return lore_terms


def build_dictionary() -> dict[str, str]:
    if not TERM_MAP_PATH.exists():
        raise FileNotFoundError(f"Source term map not found: {TERM_MAP_PATH}")

    with open(TERM_MAP_PATH, "r", encoding="utf-8") as f:
        raw_manifest = json.load(f)

    data = raw_manifest.get("data", {})
    purge_lower = {p.strip().lower() for p in PURGE_TERMS}

    new_dict: dict[str, str] = {}
    seen_lower: set[str] = set()

    # 1. Apply canonical homonym resolutions first
    for term, zh in CANONICAL_HOMONYM_RESOLUTIONS.items():
        term_clean = term.strip()
        tl = term_clean.lower()
        if not is_noise_entry(term_clean, purge_lower):
            new_dict[term_clean] = zh
            seen_lower.add(tl)

    # 2. Incorporate curated terms from terms.csv (NPCs, places, events, factions have priority over raw manifest!)
    if TERMS_CSV_PATH.exists():
        with open(TERMS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                en = (row.get("src") or "").strip()
                zh = (row.get("tgt") or "").strip()
                if not en or not zh:
                    continue
                if is_noise_entry(en, purge_lower):
                    continue
                enl = en.lower()
                if enl not in seen_lower:
                    new_dict[en] = zh
                    seen_lower.add(enl)

    # 3. Iterate through categories in domain-aware priority order
    for cat in CATEGORY_PRIORITY:
        entries = data.get(cat, {})
        if isinstance(entries, dict):
            for item in entries.values():
                if isinstance(item, dict):
                    en = str(item.get("en") or "").strip()
                    zh = str(item.get("zh-chs") or "").strip()
                    if not en or not zh:
                        continue
                    if is_noise_entry(en, purge_lower):
                        continue
                    enl = en.lower()
                    if enl not in seen_lower:
                        new_dict[en] = zh
                        seen_lower.add(enl)

    # 4. Incorporate non-noisy Lore Book titles, Entry titles, and Speaker attributions
    lore_terms = extract_lore_terms(purge_lower)
    for en, zh in lore_terms.items():
        en_clean = en.strip()
        zh_clean = zh.strip()
        if not en_clean or not zh_clean:
            continue
        if is_noise_entry(en_clean, purge_lower):
            continue
        enl = en_clean.lower()
        if enl not in seen_lower:
            new_dict[en_clean] = zh_clean
            seen_lower.add(enl)

    return new_dict


def main() -> None:
    cleaned = build_dictionary()
    OUTPUT_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DICT_PATH.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Successfully generated {OUTPUT_DICT_PATH} with {len(cleaned)} terms.")


if __name__ == "__main__":
    main()
