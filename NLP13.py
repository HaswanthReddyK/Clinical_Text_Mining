import csv
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
# ── Optional dependency: symspellpy ──────────────────────────────────────────
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
# =============================================================================
# STEP 0 ─ CONDITIONS DICTIONARY
# Note: ICD-10 codes are NO LONGER hardcoded here. Each condition declares
# the clinical concepts it cares about; ICD10Loader resolves them to actual
# CMS codes at runtime.
# =============================================================================
CONDITIONS_CONFIG = {
    "ESRD": {
        "description": "End-Stage Renal Disease",
        "keywords": [
            "esrd",
            "end stage renal disease",
            "end-stage renal disease",
            "end stage renal",
            "chronic renal failure",
            "chronic kidney failure",
            "chronic kidney disease",
            "renal failure",
            "kidney failure",
            "dialysis",
            "hemodialysis",
            "peritoneal dialysis",
            "renal transplant",
            "kidney transplant",
            "egfr",
            "gfr",
            "glomerular filtration rate",
            "stage 5 ckd",
            "ckd stage 5",
        ],
        "phrase_patterns": [
            r"chronic\s+renal\s+failure",
            r"end[\s\-]+stage\s+renal",
            r"kidney\s+disease\s+stage\s+[45]",
            r"gfr\s+result",
            r"egfr\s+result",
            r"glomerular\s+filtration",
            r"dialysis",
            r"renal\s+transplant",
            r"kidney\s+transplant",
        ],
        # Phrases used to LOOK UP ICD-10 codes from the CMS dataset.
        "icd10_search_terms": [
            "end stage renal disease",
            "chronic kidney disease with stage 5",
            "dependence on renal dialysis",
            "kidney transplant",
        ],
        "keyword_score": 1.0,
        "phrase_score": 2.0,
        "icd10_score": 3.0,
    },
    "HOSPICE_PALLIATIVE": {
        "description": "Hospice Enrollment / Palliative Status",
        "keywords": [
            "hospice",
            "palliative",
            "comfort care",
            "terminal illness",
            "terminal care",
            "end of life",
            "end-of-life",
            "dnr",
            "do not resuscitate",
            "comfort measures",
            "palliative care",
            "hospice care",
            "hospice enrollment",
        ],
        "phrase_patterns": [
            r"hospice\s+(care|enrollment|status|patient)",
            r"palliative\s+(care|status|treatment|intent)",
            r"comfort\s+(care|measures|only)",
            r"end[\s\-]+of[\s\-]+life",
            r"do\s+not\s+resuscitate",
        ],
        "icd10_search_terms": [
            "encounter for palliative care",
            "do not resuscitate",
        ],
        "keyword_score": 1.0,
        "phrase_score": 2.0,
        "icd10_score": 3.0,
    },
}
CONFIDENCE_THRESHOLD = 1.0
# =============================================================================
# STEP 1 ─ STOP-WORDS (unchanged)
# =============================================================================
STOP_WORDS = {
    # English
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "as", "not", "no", "so", "if", "then", "than", "when",
    "where", "which", "who", "what", "how", "all", "any", "both", "each",
    "more", "most", "other", "some", "such", "only", "own", "same", "also",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "up", "about", "per", "am", "i", "we", "you", "he",
    "she", "they",
    # Clinical noise
    "patient", "patients", "report", "noted", "identified", "electronically",
    "signed", "please", "see", "review", "impression", "findings", "normal",
    "within", "none", "negative", "positive", "reported", "given",
    "following", "formatting", "comment", "entered", "specimen", "type",
    "blood", "serum", "performed", "performing", "site", "date", "time",
    "pm", "md", "dr", "np", "rn", "dictated", "transcribed",
}
# =============================================================================
# NEW ─ ICD-10 LOADER (CMS source, dynamic)
# =============================================================================
ICD10_DOWNLOAD_URLS = [
    # CMS (official)
    "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip",
    "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip",
    "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip",
    "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order.zip",
    # CDC NCHS mirror (same content, more stable URL pattern)
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/icd10cm-Code%20Descriptions-2026.zip",
    "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2025/icd10cm-Code%20Descriptions-2025.zip",
]
CACHE_DIR = Path.home() / ".kp_text_mining_cache"
CACHE_DIR.mkdir(exist_ok=True)
ICD10_CACHE_FILE = CACHE_DIR / "icd10cm_codes.txt"

class ICD10Loader:
    """
    Loads the ICD-10-CM code list from CMS at runtime.
    Behaviour
    ---------
    1. If a parsed cache exists locally, use it.
    2. Otherwise download the ZIP from CMS (or CDC mirror), extract the
    'icd10cm-codes-YYYY.txt' file, parse it, and cache the parsed text.
    3. Build:
    self.codes : {'N18.6': 'End stage renal disease', ...}
    self.normalized_codes : {'N186': 'N18.6', ...}
    self.keyword_to_codes : {'dialysis': {'Z99.2', ...}, ...}
    The keyword index lets us map a free-text diagnosis phrase to the
    ICD-10 codes whose descriptions contain those words.
    """
    # Words inside ICD descriptions that are too generic to index on.
    _DESC_STOP = {
        "of", "the", "and", "in", "with", "to", "for", "due", "or", "a",
        "an", "by", "on", "at", "from", "as", "not", "other", "specified",
        "unspecified", "without", "any", "all", "type", "care", "stage",
        "disease", "disorder", "condition", "encounter", "history", "use",
    }

    def __init__(self, cache_file=ICD10_CACHE_FILE, urls=None):
        self.cache_file = Path(cache_file)
        self.urls = urls or ICD10_DOWNLOAD_URLS
        self.codes = {}
        self.normalized_codes = {}
        self.keyword_to_codes = defaultdict(set)

    # ── Public API ─────────────────────────────────────────────────────────
    def load(self, force_download=False):
        if self.cache_file.exists() and not force_download:
            print(f"[ICD-10] Using cache: {self.cache_file}")
            self._parse_file(self.cache_file)
        else:
            self._download_from_cms()
        if not self.codes:
            print("[ICD-10] WARNING: no codes parsed; using minimal fallback.")
            self._load_fallback()
        self._build_keyword_index()
        print(
            f"[ICD-10] Ready: {len(self.codes):,} codes, "
            f"{len(self.keyword_to_codes):,} keywords indexed"
        )
        return self

    def lookup(self, code):
        """Look up a code with or without the dot."""
        code = code.upper().strip()
        if code in self.codes:
            return self.codes[code]
        flat = code.replace(".", "")
        canonical = self.normalized_codes.get(flat)
        return self.codes.get(canonical) if canonical else None

    def find_codes_for_phrase(self, phrase, min_overlap=2, top_n=10):
        """
        Score each known ICD-10 code by how many words from `phrase` appear
        in its description. Returns [(code, description, score), ...]
        sorted descending.
        """
        phrase_words = {
            w for w in re.findall(r"[a-z]+", phrase.lower())
            if len(w) > 2 and w not in self._DESC_STOP
        }
        if not phrase_words:
            return []
        scores = Counter()
        for word in phrase_words:
            for code in self.keyword_to_codes.get(word, ()):
                scores[code] += 1
        return [
            (code, self.codes[code], score)
            for code, score in scores.most_common(top_n)
            if score >= min_overlap
        ]

    def find_codes_in_text(self, text, min_overlap=3, top_n=15):
        """
        Scan the entire normalised text for ICD-10 code mentions
        (literal codes like 'N18.6') AND for description-word matches.
        Returns a de-duplicated list of (code, description, score, mode).
        """
        results = {}
        # 1) Literal code mentions (e.g. "N18.6", "n186", "Z51.5")
        for m in re.finditer(r"\b([A-Z][0-9][A-Z0-9]{1,5})\b", text.upper()):
            raw = m.group(1)
            canonical = (
                raw if raw in self.codes
                else self.normalized_codes.get(raw.replace(".", ""))
            )
            if canonical:
                results[canonical] = (
                    canonical, self.codes[canonical], 99, "literal"
                )
        # 2) Description-word matches (semantic)
        for code, desc, score in self.find_codes_for_phrase(
            text, min_overlap=min_overlap, top_n=top_n
        ):
            if code not in results:
                results[code] = (code, desc, score, "semantic")
        return sorted(results.values(), key=lambda x: -x[2])

    # ── Internals ──────────────────────────────────────────────────────────
    def _download_from_cms(self):
        for url in self.urls:
            try:
                print(f"[ICD-10] Downloading: {url}")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 KP-Pipeline"},
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = resp.read()
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    target = self._pick_codes_file(zf.namelist())
                    if not target:
                        print(f"[ICD-10] No codes file inside ZIP at {url}")
                        continue
                    with zf.open(target) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    self.cache_file.write_text(content, encoding="utf-8")
                    print(f"[ICD-10] Cached → {self.cache_file}")
                    self._parse_text(content)
                    return
            except (urllib.error.URLError, zipfile.BadZipFile, OSError) as e:
                print(f"[ICD-10] Failed ({type(e).__name__}): {e}")
                continue
        print("[ICD-10] All download URLs unreachable.")

    @staticmethod
    def _pick_codes_file(names):
        # Prefer the small "codes" file (CODE<TAB>DESC); fall back to "order".
        for n in names:
            low = n.lower()
            if "icd10cm-codes" in low and low.endswith(".txt"):
                return n
        for n in names:
            low = n.lower()
            if "icd10cm-order" in low and low.endswith(".txt"):
                return n
        return None

    def _parse_file(self, path):
        with open(path, encoding="utf-8", errors="replace") as f:
            self._parse_text(f.read())

    def _parse_text(self, content):
        """
        Supports both CMS file formats:
        - icd10cm-codes-YYYY.txt
        A000 Cholera due to Vibrio cholerae 01, biovar cholerae
        - icd10cm-order-YYYY.txt (fixed-width)
        00001 A000 0 Cholera due to Vibrio ... short desc...
        """
        re_codes = re.compile(r"^([A-TV-Z][0-9][A-Z0-9]{1,5})\s+(.+)$")
        re_order = re.compile(
            r"^\d{5}\s+([A-TV-Z][0-9][A-Z0-9]{1,5})\s+\d\s+(.+?)"
            r"(?:\s{2,}.*)?$"
        )
        for line in content.splitlines():
            line = line.rstrip()
            if not line:
                continue
            m = re_codes.match(line.strip()) or re_order.match(line)
            if not m:
                continue
            code = m.group(1).upper()
            desc = m.group(2).strip()
            # Re-insert the dot after the 3rd character: "N186" → "N18.6"
            if "." not in code and len(code) > 3:
                code = f"{code[:3]}.{code[3:]}"
            self.codes[code] = desc
            self.normalized_codes[code.replace(".", "")] = code

    def _load_fallback(self):
        """Tiny built-in list — only used when network is unavailable."""
        fallback = {
            "N18.6": "End stage renal disease",
            "N18.5": "Chronic kidney disease, stage 5",
            "N18.4": "Chronic kidney disease, stage 4",
            "N18.3": "Chronic kidney disease, stage 3 (moderate)",
            "Z99.2": "Dependence on renal dialysis",
            "Z94.0": "Kidney transplant status",
            "Z51.5": "Encounter for palliative care",
            "Z66": "Do not resuscitate",
        }
        self.codes.update(fallback)
        for c in fallback:
            self.normalized_codes[c.replace(".", "")] = c

    def _build_keyword_index(self):
        for code, desc in self.codes.items():
            for word in re.findall(r"[a-z]+", desc.lower()):
                if len(word) > 3 and word not in self._DESC_STOP:
                    self.keyword_to_codes[word].add(code)


# =============================================================================
# NEW ─ SPELL CORRECTOR (SymSpell + protected clinical vocabulary)
# =============================================================================
class SpellCorrector:
    """
    • A protected clinical vocabulary is added to the SymSpell dictionary
    with a high frequency so that medical terms cannot be "corrected"
    into nearby English words (e.g. 'esrd' → 'erred').
    • Tokens that look like ICD codes, numerics, or contain digits/dots/
    hyphens are left untouched.
    • Tokens shorter than 4 characters are not corrected (too ambiguous).
    """
    CLINICAL_TERMS = [
        # Renal / ESRD
        "esrd", "ckd", "gfr", "egfr", "hemodialysis", "peritoneal",
        "dialysis", "renal", "creatinine", "glomerular", "filtration",
        "transplant", "nephrology", "uremia", "azotemia", "proteinuria",
        "albuminuria", "hyperkalemia", "nephropathy", "nephritis",
        # Hospice / Palliative
        "palliative", "hospice", "comfort", "terminal", "resuscitate",
        "dnr",
        # Common clinical terms (avoid noisy corrections)
        "hypertension", "diabetes", "myocardial", "infarction",
        "cerebrovascular", "ischemia", "pulmonary", "obstructive",
        "respiratory", "cardiac", "hepatic", "pancreatic", "neoplasm",
        "metastasis", "carcinoma", "lymphoma", "leukemia", "anemia",
        "thrombosis", "embolism", "sepsis", "pneumonia", "bronchitis",
        "edema", "effusion", "hemidiaphragm", "pleural",
    ]

    def __init__(self, max_edit_distance=2, prefix_length=7):
        self.available = SYMSPELL_AVAILABLE
        self.sym = None
        if not self.available:
            print("[SPELL] symspellpy not installed — corrector disabled.")
            print(" Install with: pip install symspellpy")
            return
        self.sym = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=prefix_length,
        )
        self._loaded = self._load_default_dictionary()
        if self._loaded:
            self._add_clinical_terms()
        else:
            self.available = False

    def _load_default_dictionary(self):
        """Locate and load symspellpy's bundled English frequency dict."""
        try:
            import importlib.resources as ir
            try:
                # Python ≥ 3.9
                ref = ir.files("symspellpy").joinpath(
                    "frequency_dictionary_en_82_765.txt"
                )
                with ir.as_file(ref) as p:
                    self.sym.load_dictionary(
                        str(p), term_index=0, count_index=1
                    )
                print(f"[SPELL] Loaded base dictionary: {p}")
                return True
            except (AttributeError, TypeError):
                pass
            # Older Python: pkg_resources
            import pkg_resources
            path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym.load_dictionary(path, term_index=0, count_index=1)
            print(f"[SPELL] Loaded base dictionary: {path}")
            return True
        except Exception as e:
            print(f"[SPELL] Failed to load base dictionary: {e}")
            return False

    def _add_clinical_terms(self):
        # Massive frequency so SymSpell prefers these over near-miss words.
        for term in self.CLINICAL_TERMS:
            self.sym.create_dictionary_entry(term, 10_000_000)

    def correct_word(self, word):
        if not self.available or not word:
            return word
        if (len(word) < 4
                or word.isdigit()
                or any(ch.isdigit() for ch in word)
                or "." in word
                or "-" in word):
            return word
        suggestions = self.sym.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True,
        )
        if suggestions and suggestions[0].distance > 0:
            return suggestions[0].term
        return word

    def correct_text(self, text):
        if not self.available or not text:
            return text
        return re.sub(
            r"\b[a-z]+\b",
            lambda m: self.correct_word(m.group(0)),
            text,
        )


# =============================================================================
# STEP 2 ─ WORD NORMALIZER (now optionally backed by SpellCorrector)
# =============================================================================
class WordNormalizer:
    """
    Lower-cases, expands medical abbreviations, optionally runs spell
    correction, and strips punctuation/noise. Behaviour is unchanged
    when no spell corrector is supplied.
    """
    _abbrev = {
        r"\besrd\b": "end stage renal disease",
        r"\bckd\b": "chronic kidney disease",
        r"\bgfr\b": "glomerular filtration rate",
        r"\begfr\b": "estimated glomerular filtration rate",
        r"\bdnr\b": "do not resuscitate",
        r"\bhd\b": "hemodialysis",
        r"\bpd\b": "peritoneal dialysis",
        r"\bhtn\b": "hypertension",
        r"\bdm\b": "diabetes mellitus",
        r"\bchf\b": "congestive heart failure",
        r"\bcopd\b": "chronic obstructive pulmonary disease",
        r"\bcva\b": "cerebrovascular accident",
        r"\bmi\b": "myocardial infarction",
    }

    def __init__(self, spell_corrector=None):
        self.spell_corrector = spell_corrector

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        t = text.lower()
        # Optional spelling correction (clinical-vocab aware).
        if self.spell_corrector is not None:
            t = self.spell_corrector.correct_text(t)

        for pattern, expansion in self._abbrev.items():
            t = re.sub(pattern, expansion, t)
        # Keep alphanumerics + spaces + dots (ICD codes) + hyphens.
        t = re.sub(r"[^a-z0-9\s\.\-]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


# =============================================================================
# STEP 3 ─ TOKENIZER 
# =============================================================================
class Tokenizer:
    def tokenize(self, text: str) -> list:
        return re.findall(r"\b[a-z][a-z0-9\-\.]*\b", text)

    def add_bigrams(self, tokens: list) -> list:
        bigrams = [
            f"{tokens[i]}__{tokens[i + 1]}"
            for i in range(len(tokens) - 1)
        ]
        return tokens + bigrams


# =============================================================================
# STEP 4 ─ STOP-WORD REMOVER 
# =============================================================================
class StopWordRemover:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or STOP_WORDS

    def remove(self, tokens: list) -> list:
        return [
            t for t in tokens
            if t not in self.stop_words
            and len(t) > 2
            and not t.isdigit()
        ]


# =============================================================================
# STEP 5 ─ MEDICAL STEMMER (unchanged)
# =============================================================================
class MedicalStemmer:
    _suffixes = [
        "ization", "isation", "ational", "ation", "ness", "ment",
        "ities", "ity", "ying", "ing", "tion", "ies", "ed", "ly",
        "al", "s",
    ]
    def stem(self, word: str) -> str:
        if len(word) <= 4 or re.match(r"[a-z]\d{2}", word):
            return word
        for suffix in self._suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) > 3:
                return word[:-len(suffix)]
        return word

    def stem_all(self, tokens: list) -> list:
        return [self.stem(t) for t in tokens]


# =============================================================================
# STEP 6 ─ INVERTED INDEX 
# =============================================================================
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.doc_freq = defaultdict(int)
        self.doc_count = 0

    def add_document(self, doc_id: str, tokens: list):
        self.doc_count += 1
        seen = set()
        for tok in tokens:
            self.index[tok].add(doc_id)
            if tok not in seen:
                self.doc_freq[tok] += 1
                seen.add(tok)

    def lookup(self, term: str) -> set:
        return self.index.get(term, set())

    def phrase_lookup(self, terms: list) -> set:
        if not terms:
            return set()
        result = self.lookup(terms[0])
        for t in terms[1:]:
            result = result & self.lookup(t)
        return result

    def tfidf(self, term: str, doc_id: str, tf: float) -> float:
        df = self.doc_freq.get(term, 1)
        idf = math.log((self.doc_count + 1) / (df + 1))
        return round(tf * idf, 4)

    def top_terms(self, n=20) -> list:
        return sorted(self.doc_freq.items(), key=lambda x: -x[1])[:n]


# =============================================================================
# STEP 7 ─ CONDITION DETECTOR 
# =============================================================================
class ConditionDetector:
    """
    Scores each document per condition using:
    A. ICD-10 codes resolved DYNAMICALLY from the CMS dataset
    B. Keyword lookup via inverted index
    C. Regex phrase patterns
    """
    def __init__(self, conditions_config, inv_index, normalizer, tokenizer,
                 stop_remover, stemmer, icd_loader):
        self.cfg = conditions_config
        self.idx = inv_index
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.stop_remover = stop_remover
        self.stemmer = stemmer
        self.icd_loader = icd_loader
        # Pre-resolve "icd10_search_terms" → actual CMS codes once at init
        self._resolved_codes = self._resolve_condition_codes()

    def _resolve_condition_codes(self):
        """For every condition, find the CMS codes that match its concepts."""
        resolved = {}
        for cond, cfg in self.cfg.items():
            code_set = {}
            for term in cfg.get("icd10_search_terms", []):
                for code, desc, _score in self.icd_loader.find_codes_for_phrase(
                    term, min_overlap=2, top_n=8
                ):
                    code_set[code] = desc
            resolved[cond] = code_set
            print(
                f"[DETECTOR] {cond}: resolved {len(code_set)} ICD-10 codes "
                f"from CMS — examples: {list(code_set.keys())[:5]}"
            )
        return resolved

    def _keyword_tokens(self, keyword: str) -> list:
        n = self.normalizer.normalize(keyword)
        t = self.tokenizer.tokenize(n)
        c = self.stop_remover.remove(t)
        return self.stemmer.stem_all(c)

    def detect(self, doc_id: str, raw_text: str) -> dict:
        norm_text = self.normalizer.normalize(raw_text)
        results = {}
        for condition, cfg in self.cfg.items():
            matched_keywords = []
            matched_patterns = []
            matched_icd = []
            score = 0.0
            # ── A. ICD-10 detection (CMS-resolved + literal in text) ──────
            cms_codes = self._resolved_codes.get(condition, {})
            text_no_dot = norm_text.replace(".", "").upper()
            for code, desc in cms_codes.items():
                flat = code.replace(".", "").upper()
                if flat in text_no_dot:
                    matched_icd.append({"code": code, "description": desc})
                    score += cfg["icd10_score"]
            # Also pick up any literal ICD codes the text mentions that
            # match the codes resolved for this condition.
            for code, desc, _, mode in self.icd_loader.find_codes_in_text(
                norm_text, min_overlap=4, top_n=5
            ):
                if code in cms_codes and not any(
                    m["code"] == code for m in matched_icd
                ):
                    matched_icd.append({"code": code, "description": desc})
                    score += cfg["icd10_score"]
            # ── B. Keyword lookup via inverted index ──────────────────────
            for keyword in cfg["keywords"]:
                kw_tokens = self._keyword_tokens(keyword)
                if not kw_tokens:
                    continue
                if len(kw_tokens) == 1:
                    if doc_id in self.idx.lookup(kw_tokens[0]):
                        matched_keywords.append(keyword)
                        score += cfg["keyword_score"]
                else:
                    if doc_id in self.idx.phrase_lookup(kw_tokens):
                        matched_keywords.append(keyword)
                        score += cfg["keyword_score"] * 1.5
            # ── C. Regex phrase patterns ──────────────────────────────────
            for pattern in cfg["phrase_patterns"]:
                m = re.search(pattern, norm_text)
                if m:
                    matched_patterns.append(
                        {"pattern": pattern, "match": m.group(0)}
                    )
                    score += cfg["phrase_score"]
            results[condition] = {
                "detected": score >= CONFIDENCE_THRESHOLD,
                "confidence_score": round(score, 2),
                "matched_keywords": list(set(matched_keywords)),
                "matched_patterns": matched_patterns,
                "matched_icd10": matched_icd,
                "description": cfg["description"],
            }
        return results


# =============================================================================
# STEP 8 ─ MAIN PIPELINE ORCHESTRATOR
# =============================================================================
class TextMiningPipeline:
    """
    Phase 1 — build_corpus_index(): preprocess every record, populate index.
    Phase 2 — detect_conditions(): walk the corpus, run ConditionDetector.
    """
    def __init__(self, enable_spell_correction=True,
                 force_icd_download=False):
        # Load CMS ICD-10 codes ONCE at startup
        self.icd_loader = ICD10Loader().load(force_download=force_icd_download)
        # Spell corrector (optional)
        self.spell_corrector = (
            SpellCorrector() if enable_spell_correction else None
        )
        self.normalizer = WordNormalizer(spell_corrector=self.spell_corrector)
        self.tokenizer = Tokenizer()
        self.stop_remover = StopWordRemover()
        self.stemmer = MedicalStemmer()
        self.inv_index = InvertedIndex()
        self.doc_store = {}

    # ── Phase 1 ────────────────────────────────────────────────────────────
    def build_corpus_index(self, records: list):
        print(f"\n[PHASE 1] Building inverted index for {len(records)} "
              f"records …")
        for rec in records:
            doc_id = f"{rec['DOCUMENT_ID']}_{rec.get('CONTACT_LINE', 0)}"
            raw_text = rec.get("RESULT_TEXT", "") or ""
            normalised = self.normalizer.normalize(raw_text)
            tokens = self.tokenizer.tokenize(normalised)
            with_bi = self.tokenizer.add_bigrams(tokens)
            clean = self.stop_remover.remove(with_bi)
            stemmed = self.stemmer.stem_all(clean)
            self.doc_store[doc_id] = {
                "document_id": rec["DOCUMENT_ID"],
                "contact_line": rec.get("CONTACT_LINE"),
                "contact_date": rec.get("CONTACT_DATE"),
                "raw_text": raw_text,
            }
            self.inv_index.add_document(doc_id, stemmed)
        print(f" → Index size : {len(self.inv_index.index):,} unique terms")
        print(f" → Documents : {self.inv_index.doc_count:,}")
        return self

    # ── Phase 2 ────────────────────────────────────────────────────────────
    def detect_conditions(self) -> list:
        print("\n[PHASE 2] Detecting conditions …")
        detector = ConditionDetector(
            CONDITIONS_CONFIG,
            self.inv_index,
            self.normalizer,
            self.tokenizer,
            self.stop_remover,
            self.stemmer,
            self.icd_loader,
        )
        results = []
        for doc_id, meta in self.doc_store.items():
            detections = detector.detect(doc_id, meta["raw_text"])
            if any(v["detected"] for v in detections.values()):
                results.append({
                    "doc_id": doc_id,
                    "document_id": meta["document_id"],
                    "contact_line": meta["contact_line"],
                    "contact_date": str(meta["contact_date"]),
                    "text_snippet": meta["raw_text"][:250],
                    "conditions": detections,
                })
        print(f" → Flagged documents : {len(results)}")
        return results

    def _reset_index(self):
        self.inv_index = InvertedIndex()
        self.doc_store = {}

    def run(self, records: list) -> list:
        self.build_corpus_index(records)
        return self.detect_conditions()

    def run_batched(self, record_iter, batch_size=50_000,
                    output_csv="text_mining_results.csv"):
        """
        Stream millions of records without loading everything into memory.
        Processes in chunks: index → detect → write CSV → clear → repeat.
        Accepts any iterable of record dicts (list, generator, DB cursor).
        """
        import csv as _csv
        fieldnames = [
            "DOCUMENT_ID", "CONTACT_LINE", "CONTACT_DATE",
            "CONDITION_CODE", "CONDITION_DESC", "CONFIDENCE_SCORE",
            "MATCHED_KEYWORDS", "MATCHED_PATTERNS", "MATCHED_ICD10_CMS",
            "TEXT_SNIPPET", "PROCESSED_TS",
        ]
        total_processed = 0
        total_flagged = 0
        batch = []

        def _flush(batch):
            nonlocal total_flagged
            if not batch:
                return
            self._reset_index()
            self.build_corpus_index(batch)
            results = self.detect_conditions()
            total_flagged += len(results)
            for r in results:
                for condition, det in r["conditions"].items():
                    if det["detected"]:
                        icd_codes = " | ".join(
                            f"{c['code']}:{c['description']}"
                            for c in det.get("matched_icd10", [])
                        )
                        writer.writerow({
                            "DOCUMENT_ID": r["document_id"],
                            "CONTACT_LINE": r["contact_line"],
                            "CONTACT_DATE": r["contact_date"],
                            "CONDITION_CODE": condition,
                            "CONDITION_DESC": det["description"],
                            "CONFIDENCE_SCORE": det["confidence_score"],
                            "MATCHED_KEYWORDS": " | ".join(det["matched_keywords"]),
                            "MATCHED_PATTERNS": " | ".join(
                                p["match"] for p in det["matched_patterns"]
                            ),
                            "MATCHED_ICD10_CMS": icd_codes,
                            "TEXT_SNIPPET": r["text_snippet"][:200],
                            "PROCESSED_TS": datetime.now().isoformat(),
                        })
            f.flush()

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in record_iter:
                batch.append(rec)
                total_processed += 1
                if len(batch) >= batch_size:
                    print(f"[BATCH] {total_processed:,} records processed …")
                    _flush(batch)
                    batch = []
            _flush(batch)

        print(f"[DONE] Processed: {total_processed:,} | Flagged: {total_flagged:,}")
        print(f"[DONE] Output → {output_csv}")
        return total_processed, total_flagged

    def index_stats(self) -> dict:
        return {
            "total_documents": self.inv_index.doc_count,
            "unique_terms": len(self.inv_index.index),
            "top_20_terms": self.inv_index.top_terms(20),
            "icd10_codes_loaded": len(self.icd_loader.codes),
        }


# =============================================================================
# STEP 9 ─ DATABASE EXTRACTION (Databricks)
# =============================================================================

_DATABRICKS_SQL_TEMPLATE = """
    SELECT
    DOCUMENT_ID,
    CONTACT_LINE,
    CAST(CONTACT_DATE AS STRING) AS CONTACT_DATE,
    RESULT_TEXT_KEY,
    RESULT_TEXT
    FROM {table_ref}
    WHERE RESULT_TEXT IS NOT NULL
    AND TRIM(RESULT_TEXT) <> ''
    ORDER BY DOCUMENT_ID, CONTACT_LINE
    {limit_clause}
"""

def _build_databricks_query(catalog, schema, table, limit):
    """Compose the standard extraction query. catalog may be None
    (hive_metastore-style two-part names)."""
    table_ref = f"{schema}.{table}" if not catalog else \
        f"{catalog}.{schema}.{table}"
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    return _DATABRICKS_SQL_TEMPLATE.format(
        table_ref=table_ref, limit_clause=limit_clause
    )


def stream_from_databricks(
    server_hostname,
    http_path,
    access_token,
    catalog="mtp_psup",
    schema="mtp_dsw",
    table="docs_rcvd_rslt_texts_ma",
    limit=None,
    batch_size=10_000,
):
    """
    Generator: yields one record dict at a time — never loads all rows into
    memory. Feed directly into pipeline.run_batched() for scale.

    Connection details: SQL Warehouses → <warehouse> → Connection Details
    Auth: PAT token or OAuth M2M (read from env, not hardcoded):
        token = os.environ["DATABRICKS_TOKEN"]
    """
    try:
        from databricks import sql
    except ImportError:
        print("[DB ERROR] databricks-sql-connector not installed. "
              "Run: pip install databricks-sql-connector")
        return
    query = _build_databricks_query(catalog, schema, table, limit)
    try:
        with sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                cols = [d[0] for d in cursor.description]
                fetched = 0
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    fetched += len(rows)
                    print(f" [Databricks] streamed {fetched:,} rows …")
                    for row in rows:
                        yield dict(zip(cols, row))
    except Exception as e:
        print(f"[DB ERROR] {e}")


# =============================================================================
# STEP 10 ─ EXPORTERS
# =============================================================================
def export_csv(results, filepath="text_mining_results.csv"):
    rows = []
    for r in results:
        for condition, det in r["conditions"].items():
            if det["detected"]:
                icd_codes = " | ".join(
                    f"{c['code']}:{c['description']}"
                    for c in det.get("matched_icd10", [])
                )
                rows.append({
                    "DOCUMENT_ID": r["document_id"],
                    "CONTACT_LINE": r["contact_line"],
                    "CONTACT_DATE": r["contact_date"],
                    "CONDITION_CODE": condition,
                    "CONDITION_DESC": det["description"],
                    "CONFIDENCE_SCORE": det["confidence_score"],
                    "MATCHED_KEYWORDS": " | ".join(det["matched_keywords"]),
                    "MATCHED_PATTERNS": " | ".join(
                        p["match"] for p in det["matched_patterns"]
                    ),
                    "MATCHED_ICD10_CMS": icd_codes,
                    "TEXT_SNIPPET": r["text_snippet"][:200],
                    "PROCESSED_TS": datetime.now().isoformat(),
                })
    if not rows:
        print("[EXPORT] No results to export.")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[EXPORT] CSV written → {filepath} ({len(rows)} rows)")


def export_sql_inserts(results, filepath="text_mining_inserts.sql"):
    lines = [
        "-- Auto-generated by KP Text Mining Pipeline (Enhanced)",
        f"-- Generated at: {datetime.now().isoformat()}",
        "",
        "CREATE TABLE IF NOT EXISTS text_mining_condition_flags (",
        " DOCUMENT_ID BIGINT,",
        " CONTACT_LINE INT,",
        " CONDITION_CODE VARCHAR(50),",
        " CONFIDENCE_SCORE DECIMAL(8,2),",
        " MATCHED_KEYWORDS VARCHAR(500),",
        " MATCHED_ICD10 VARCHAR(500),",
        " PROCESSED_TS TIMESTAMP",
        ");",
        "",
    ]
    for r in results:
        for condition, det in r["conditions"].items():
            if det["detected"]:
                kw = " | ".join(det["matched_keywords"])[:490].replace(
                    "'", "''"
                )
                icd = " | ".join(
                    c["code"] for c in det.get("matched_icd10", [])
                )[:490].replace("'", "''")
                lines.append(
                    f"INSERT INTO text_mining_condition_flags VALUES "
                    f"({r['document_id']}, "
                    f"{r['contact_line'] or 'NULL'}, "
                    f"'{condition}', {det['confidence_score']}, "
                    f"'{kw}', '{icd}', CURRENT_TIMESTAMP);"
                )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[EXPORT] SQL written → {filepath}")


def export_json(results, filepath="text_mining_results.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[EXPORT] JSON written → {filepath}")


# =============================================================================
# STEP 11 ─ ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    SAMPLE_RECORDS = [
        {
            "DOCUMENT_ID": 1708275951,
            "CONTACT_LINE": 1,
            "CONTACT_DATE": "2026-02-10",
            # Note typos: "GFR resullts", "chronnic renal", "patiens" —
            # SymSpell + clinical vocab should clean these up.
            "RESULT_TEXT": (
                "GFR resullts reported in mL/min/1.73 sq.m. "
                "For African American individuals where race was not "
                "specified, please multiply the GFR by 1.21. "
                "This eGFR is validated for stable chronnic renal failure "
                "patiens. This equation is unreliable in acute illness."
            ),
        },
        {
            "DOCUMENT_ID": 1708275951,
            "CONTACT_LINE": 2,
            "CONTACT_DATE": "2026-02-10",
            "RESULT_TEXT": "or patients with normal  chronic renal failure.",
        },
        {
            "DOCUMENT_ID": 1704337839,
            "CONTACT_LINE": 1,
            "CONTACT_DATE": "2026-02-10",
            "RESULT_TEXT": (
                "IMPRESSION: Elevation of the right hemidiaphragm with a "
                "small pleural effusion."
            ),
        },
        {
            "DOCUMENT_ID": 1704337839,
            "CONTACT_LINE": 2,
            "CONTACT_DATE": "2026-02-10",
            "RESULT_TEXT": (
                "Dictated and electronically signed by Ivan Petrovitch, "
                "MD on 5/2."
            ),
        },
        {
            "DOCUMENT_ID": 9999000001,
            "CONTACT_LINE": 1,
            "CONTACT_DATE": "2026-02-10",

            "RESULT_TEXT": (
                "Patient has been enrolled in hospice care per family "
                "request. Comfort care measures only. Z51.5 documented."
            ),
        },
        {
            "DOCUMENT_ID": 9999000002,
            "CONTACT_LINE": 1,
            "CONTACT_DATE": "2026-02-10",
            "RESULT_TEXT": (
                "ESRD patient on hemodialysis three times weekly. "
                "Kidney transplant workup deferred. ICD N18.6."
            ),
        },
    ]
    pipeline = TextMiningPipeline(
        enable_spell_correction=True,
        force_icd_download=False,
    )
    results = pipeline.run(SAMPLE_RECORDS)
    stats = pipeline.index_stats()
    print(f"\n[INDEX STATS]")
    print(f" Total docs : {stats['total_documents']}")
    print(f" Unique terms : {stats['unique_terms']}")
    print(f" ICD-10 codes (CMS): {stats['icd10_codes_loaded']:,}")
    print(f" Top terms : "
          f"{[t for t, _ in stats['top_20_terms'][:10]]}")
    print("\n" + "=" * 70)
    print(" DETECTED CONDITIONS")
    print("=" * 70)
    for r in results:
        print(f"\n Doc ID : {r['document_id']} | Line : {r['contact_line']}")
        print(f" Text : {r['text_snippet'][:100]} …")
        for cond, det in r["conditions"].items():
            if det["detected"]:
                print(f" ► {cond} — score={det['confidence_score']}")
                print(f" Keywords : {det['matched_keywords']}")
                print(f" Patterns : "
                      f"{[p['match'] for p in det['matched_patterns']]}")
                if det.get("matched_icd10"):
                    print(f" ICD-10 : "
                          f"{[c['code'] for c in det['matched_icd10']]}")
    export_csv(results)
    export_sql_inserts(results)
    export_json(results)
    # ── To run millions of records from Databricks: ─────────────────────────
    # import os
    # pipeline = TextMiningPipeline(enable_spell_correction=False)  # faster
    # pipeline.run_batched(
    #     record_iter=stream_from_databricks(
    #         server_hostname="",
    #         http_path="",
    #         access_token=os.environ[""],
    #         catalog="mtp_psup",
    #         schema="mtp_dsw",
    #         table="docs_rcvd_rslt_texts_ma",
    #     ),
    #     batch_size=50_000,
    #     output_csv="text_mining_results.csv",
    # )
