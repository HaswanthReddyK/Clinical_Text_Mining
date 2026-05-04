#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency flags — graceful degradation when libs are missing
# ---------------------------------------------------------------------------
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    log.warning("spaCy not installed — regex-only NER. Run: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    log.warning("symspellpy not installed — spell correction disabled. Run: pip install symspellpy")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn/numpy not installed — keyword ICD fallback only. Run: pip install scikit-learn numpy")

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    log.warning("beautifulsoup4 not installed. Run: pip install beautifulsoup4")


# ---------------------------------------------------------------------------
# Dependency auto-installer (activated by --auto-install CLI flag)
# ---------------------------------------------------------------------------
def _auto_install_and_restart() -> None:
    """Install missing optional packages and restart the script."""
    pkgs: List[str] = []
    if not SPACY_AVAILABLE:
        pkgs.append("spacy")
    if not SYMSPELL_AVAILABLE:
        pkgs.append("symspellpy")
    if not SKLEARN_AVAILABLE:
        pkgs += ["scikit-learn", "numpy"]
    if not BS4_AVAILABLE:
        pkgs.append("beautifulsoup4")

    if not pkgs:
        log.info("[Setup] All optional packages already installed.")
        return

    log.info(f"[Setup] Installing: {' '.join(pkgs)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)

    if "spacy" in pkgs:
        log.info("[Setup] Downloading spaCy model en_core_web_sm …")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    log.info("[Setup] Restarting pipeline with all packages active …")
    # Replace current process — new process will import the just-installed libs
    os.execv(sys.executable, [sys.executable] + sys.argv)

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    """All tunable pipeline settings in one place."""
    # text processing
    max_text_chars: int = 50_000

    # ICD-10 mapping
    icd_top_n: int = 3
    tfidf_min_score: float = 0.05       # raised from 0.01 — reduces noise
    tfidf_max_features: int = 60_000
    tfidf_ngram_max: int = 3

    # spell checking
    spell_max_edit_distance: int = 2

    # NLP batching
    nlp_batch_size: int = 64

    # accuracy filters
    exclude_negated: bool = True
    exclude_sections: List[str] = field(
        default_factory=lambda: ["FAMILY_HISTORY", "PAST_HISTORY", "SOCIAL_HISTORY"]
    )
    exclude_historical: bool = False    # inline "history of X" entities

    # run behaviour
    batch_log_every: int = 100
    worker_threads: int = 1

    # data source
    force_icd_refresh: bool = False

    # checkpointing
    checkpoint_file: str = "pipeline_checkpoint.json"

    # output
    output_dir: str = "."

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class CheckpointManager:
    """Persists the last Databricks anchor for resumable extraction."""

    def __init__(self, filepath: str = "pipeline_checkpoint.json") -> None:
        self._path = Path(filepath)

    def save(self, anchor_value, batch_num: int, total_processed: int) -> None:
        self._path.write_text(
            json.dumps({
                "anchor_value":    str(anchor_value),
                "batch_num":       batch_num,
                "total_processed": total_processed,
                "saved_at":        datetime.now().isoformat(),
            }),
            encoding="utf-8",
        )
        log.info(f"[Checkpoint] anchor={anchor_value}, batch={batch_num}, processed={total_processed:,}")

    def load(self) -> Optional[Dict]:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            log.info(f"[Checkpoint] Resuming from anchor={data['anchor_value']}, batch={data['batch_num']}")
            return data
        except Exception as exc:
            log.warning(f"[Checkpoint] Load failed ({exc}) — starting fresh.")
            return None

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
            log.info("[Checkpoint] Cleared — extraction complete.")

# =============================================================================
# DATABRICKS CONFIGURATION  — set env vars or fill in directly
# =============================================================================
# Required environment variables (preferred) or set literals below:
#   DATABRICKS_HOST   e.g. myworkspace.azuredatabricks.net
#   DATABRICKS_TOKEN  personal access token or service principal secret
#   DATABRICKS_HTTP   e.g. /sql/1.0/warehouses/abc123
#
# Install:  pip install databricks-sql-connector
# =============================================================================

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST",  "<workspace>.azuredatabricks.net")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN",  "<token>")
DATABRICKS_HTTP  = os.getenv("DATABRICKS_HTTP",   "/sql/1.0/warehouses/<id>")
TABLE_NAME       = os.getenv("DB_TABLE",          "catalog.schema.clinical_table")
TEXT_COLUMN      = os.getenv("DB_TEXT_COLUMN",    "RESULT_TEXT")
ANCHOR_COL       = os.getenv("DB_ANCHOR_COL",     "DOCUMENT_ID")
DB_BATCH_SIZE    = int(os.getenv("DB_BATCH_SIZE", "500"))

try:
    from databricks import sql as _dbsql  # type: ignore
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

def fetch_from_databricks(
    query: str,
    *,
    batch_size: int = DB_BATCH_SIZE,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    last_anchor=None,
):
    """
    Keyset-paginated generator over a Databricks SQL warehouse.
    Yields list[dict] batches; safe to resume from a checkpoint anchor.
    Handles transient connection errors with exponential back-off.
    Scales to tens of millions of rows.
    """
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError(
            "databricks-sql-connector not installed. "
            "Run: pip install databricks-sql-connector"
        )

    connect_args = dict(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP,
        access_token=DATABRICKS_TOKEN,
    )

    while True:  # outer retry loop for connection failures
        attempt = 0
        try:
            with _dbsql.connect(**connect_args) as conn:
                with conn.cursor() as cur:
                    while True:
                        keyset = (
                            f" AND {ANCHOR_COL} > {last_anchor!r}"
                            if last_anchor is not None else ""
                        )
                        sql = (
                            f"{query}{keyset} "
                            f"ORDER BY {ANCHOR_COL} "
                            f"LIMIT {batch_size}"
                        )
                        for attempt in range(1, max_retries + 1):
                            try:
                                cur.execute(sql)
                                break
                            except Exception as exc:
                                if attempt == max_retries:
                                    raise
                                wait = retry_delay * (2 ** (attempt - 1))
                                log.warning(
                                    f"[Databricks] Query attempt {attempt}/{max_retries} "
                                    f"failed ({exc}). Retrying in {wait:.0f}s …"
                                )
                                time.sleep(wait)

                        rows = cur.fetchall()
                        if not rows:
                            return
                        cols = [d[0] for d in cur.description]
                        batch = [dict(zip(cols, r)) for r in rows]
                        last_anchor = batch[-1][ANCHOR_COL]
                        yield batch
                        if len(rows) < batch_size:
                            return   # last page
            return  # clean exit — no exception
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                log.error(f"[Databricks] Giving up after {max_retries} attempts: {exc}")
                raise
            wait = retry_delay * (2 ** (attempt - 1))
            log.warning(
                f"[Databricks] Connection error ({exc}). "
                f"Reconnecting in {wait:.0f}s (attempt {attempt}/{max_retries}) …"
            )
            time.sleep(wait)


# =============================================================================
# STEP 1 — ICD-10-CM FETCHER
# =============================================================================
class ICD10CMFetcher:
    """Downloads the full ICD-10-CM tabular-order file from CMS and caches it."""

    CACHE_FILE = Path("icd10cm_codes.json")
    CMS_PAGE_URL = "https://www.cms.gov/medicare/coding-billing/icd-10-codes"

    _FALLBACK_URLS: List[str] = [
        "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip",
        "https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip",
        "https://www.cms.gov/files/zip/2024-code-descriptions-tabular-order-updated-02012024.zip",
        "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2025/icd10cm-codes-2025.zip",
    ]

    # Skip quarterly / addenda ZIP files — they contain only deltas, not the full list.
    _QUARTERLY_RE = re.compile(r"\b(?:april|january|july|october|addenda|update[d]?)\b", re.IGNORECASE)

    def __init__(self, force_refresh: bool = False) -> None:
        self.force_refresh = force_refresh
        self.codes: Dict[str, str] = {}

    def fetch(self) -> Dict[str, str]:
        if not self.force_refresh and self.CACHE_FILE.exists():
            log.info(f"ICD-10-CM: loading from cache ({self.CACHE_FILE}) …")
            self.codes = json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
            log.info(f"  {len(self.codes):,} codes loaded.")
            return self.codes

        log.info("ICD-10-CM: fetching from CMS …")
        zip_bytes = self._scrape_cms_page() or self._try_fallbacks()
        if zip_bytes and self._is_zip(zip_bytes):
            self.codes = self._parse_zip(zip_bytes)
        if self.codes:
            self._save_cache()
        else:
            log.error("Could not retrieve ICD-10-CM codes — ICD mapping disabled.")
        return self.codes

    @staticmethod
    def _is_zip(data: bytes) -> bool:
        return data[:2] == b"PK"

    def _scrape_cms_page(self) -> Optional[bytes]:
        if not BS4_AVAILABLE:
            return None
        try:
            resp = requests.get(self.CMS_PAGE_URL, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href: str = a["href"]
                fname = href.split("/")[-1]
                if (
                    "tabular" in fname.lower()
                    and fname.lower().endswith(".zip")
                    and not self._QUARTERLY_RE.search(fname)
                ):
                    url = href if href.startswith("http") else f"https://www.cms.gov{href}"
                    log.info(f"  CMS link: {url}")
                    return self._download(url)
        except Exception as exc:
            log.warning(f"  CMS scrape failed: {exc}")
        return None

    def _try_fallbacks(self) -> Optional[bytes]:
        for url in self._FALLBACK_URLS:
            data = self._download(url)
            if data and self._is_zip(data):
                return data
        return None

    @staticmethod
    def _download(url: str) -> Optional[bytes]:
        try:
            log.info(f"  Downloading: {url}")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            log.info(f"  {len(resp.content) / 1_048_576:.1f} MB downloaded")
            return resp.content
        except Exception as exc:
            log.warning(f"  Download failed: {exc}")
        return None

    def _parse_zip(self, zip_bytes: bytes) -> Dict[str, str]:
        codes: Dict[str, str] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Prefer the fixed-width tabular-order file; skip addenda
                target = next(
                    (n for n in zf.namelist()
                     if n.lower().endswith(".txt")
                     and "order" in n.lower()
                     and "addenda" not in n.lower()),
                    None,
                ) or next(
                    (n for n in zf.namelist()
                     if n.lower().endswith(".txt")
                     and "icd10cm" in n.lower()
                     and "addenda" not in n.lower()),
                    None,
                )
                if not target:
                    log.error("  No suitable .txt file found in ZIP.")
                    return codes

                log.info(f"  Parsing: {target}")
                with zf.open(target) as fh:
                    for raw in fh:
                        line = raw.decode("latin-1", errors="replace").rstrip("\r\n")
                        if len(line) < 14:
                            continue
                        code = line[6:13].strip()
                        flag = line[13].strip()
                        if flag == "1" or not code:
                            continue
                        short = line[15:75].strip() if len(line) > 75 else line[15:].strip()
                        long_ = line[76:].strip() if len(line) > 76 else short
                        codes[code] = long_ or short

                log.info(f"  Parsed {len(codes):,} billable codes.")
        except Exception as exc:
            log.error(f"  ZIP parse error: {exc}")
        return codes

    def _save_cache(self) -> None:
        self.CACHE_FILE.write_text(json.dumps(self.codes, ensure_ascii=False), encoding="utf-8")
        log.info(f"  Cache -> {self.CACHE_FILE}")

# =============================================================================
# STEP 2 — CLINICAL SPELL CHECKER
# =============================================================================
class ClinicalSpellChecker:
    """SymSpell corrector that protects medical vocabulary from alteration."""

    PROTECTED: frozenset = frozenset({
        "esrd", "ckd", "egfr", "gfr", "dnr", "htn", "chf", "copd",
        "cva", "bun", "icd", "cpt", "hcpcs", "hd", "pd", "np", "rn",
        "hemodialysis", "peritoneal", "dialysis", "glomerular", "filtration",
        "renal", "nephropathy", "proteinuria", "creatinine", "phosphorus",
        "potassium", "hemoglobin", "erythropoietin", "ferritin", "albumin",
        "hospice", "palliative", "resuscitate", "prognosis", "metastatic",
        "oncology", "carcinoma", "lymphoma", "myocardial", "infarction",
        "cerebrovascular", "hypertension", "hypotension", "tachycardia",
        "bradycardia", "dyspnea", "dysphagia", "nausea", "emesis",
        "opioid", "morphine", "fentanyl", "azotemia", "uremia",
        "nephrotic", "glomerulonephritis", "nephrology", "urology",
        # <- ADD HERE (3/3): new condition abbreviations / terms
    })

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._sym: Optional[SymSpell] = None
        if SYMSPELL_AVAILABLE:
            self._init_symspell()

    def _init_symspell(self) -> None:
        self._sym = SymSpell(
            max_dictionary_edit_distance=self._config.spell_max_edit_distance,
            prefix_length=7,
        )
        dict_path = self._find_dict()
        if dict_path:
            self._sym.load_dictionary(dict_path, term_index=0, count_index=1)
            log.info(f"SymSpell dictionary: {dict_path}")
        else:
            log.warning("SymSpell frequency dictionary not found.")
        for term in self.PROTECTED:
            self._sym.create_dictionary_entry(term, 10_000_000)

    @staticmethod
    def _find_dict() -> Optional[str]:
        try:
            import symspellpy as _sp
            candidates = list(Path(_sp.__file__).parent.glob("frequency_dictionary_en*.txt"))
            return str(candidates[0]) if candidates else None
        except Exception:
            return None

    def correct(self, text: str) -> Tuple[str, List[Dict]]:
        if not self._sym:
            return text, []
        corrections: List[Dict] = []
        corrected = text
        for m in re.finditer(r"\b[a-zA-Z]{4,}\b", text):
            word = m.group()
            lower = word.lower()
            if lower in self.PROTECTED or re.match(r"[a-z]\d", lower):
                continue
            hits = self._sym.lookup(
                lower, Verbosity.CLOSEST,
                max_edit_distance=self._config.spell_max_edit_distance,
            )
            if hits and hits[0].term != lower and hits[0].distance >= 1:
                fixed = hits[0].term
                corrections.append({"original": word, "corrected": fixed, "distance": hits[0].distance})
                corrected = re.sub(rf"\b{re.escape(word)}\b", fixed, corrected, count=1)
        return corrected, corrections

# =============================================================================
# STEP 3 — MEDICAL ENTITY EXTRACTOR
# =============================================================================
class MedicalEntityExtractor:
    """
    spaCy NER + custom EntityRuler + regex fallback with:
      - Pre-compiled patterns (v2)
      - Clause-aware negation detection (v2)
      - Clinical section context (v2)
      - Temporal qualifier detection  [NEW v3]
      - CKD stage extraction          [NEW v3]
      - eGFR value extraction         [NEW v3]

    Extension:
      <- ADD HERE (1/3): _RULER_ENTRIES for new conditions
      <- ADD HERE (2/3): _REGEX for new conditions
    """

    # ── spaCy patterns ───────────────────────────────────────────────────────
    # <- ADD HERE (1/3)
    _RULER_ENTRIES: List[Tuple[str, List[str]]] = [
        ("KIDNEY_DISEASE", [
            "end stage renal disease", "end-stage renal disease",
            "chronic kidney disease", "chronic renal failure",
            "ESRD", "CKD", "renal failure", "kidney failure",
            "end stage renal", "stage 5 ckd", "ckd stage 5",
        ]),
        ("DIALYSIS", ["hemodialysis", "peritoneal dialysis", "renal dialysis", "dialysis"]),
        ("TRANSPLANT", ["kidney transplant", "renal transplant"]),
        ("KIDNEY_FUNCTION", [
            "glomerular filtration rate", "estimated glomerular filtration rate",
            "GFR", "eGFR", "creatinine", "egfr result", "gfr result",
        ]),
        ("HOSPICE", ["hospice care", "hospice enrollment", "hospice patient", "hospice status", "hospice"]),
        ("PALLIATIVE", ["palliative care", "palliative status", "palliative treatment", "palliative intent", "palliative"]),
        ("COMFORT_CARE", ["comfort care", "comfort measures", "comfort measures only", "comfort care only"]),
        ("END_OF_LIFE", ["end of life", "end-of-life", "do not resuscitate", "DNR", "terminal illness", "terminal care"]),
        ("HEART_DISEASE", ["congestive heart failure", "CHF", "myocardial infarction", "atrial fibrillation", "heart failure"]),
        ("LUNG_DISEASE", ["chronic obstructive pulmonary disease", "COPD", "pulmonary embolism", "pulmonary fibrosis"]),
        ("DIABETES", ["diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "insulin dependent diabetes"]),
        ("CANCER", ["malignant neoplasm", "metastatic cancer", "carcinoma", "lymphoma", "leukemia"]),
        ("STROKE", ["cerebrovascular accident", "CVA", "ischemic stroke", "hemorrhagic stroke",
                   "transient ischemic attack", "TIA"]),
        ("HYPERTENSION", ["hypertension", "hypertensive", "high blood pressure", "HTN"]),
    ]

    # ── Regex fallback patterns ──────────────────────────────────────────────
    # <- ADD HERE (2/3)
    _REGEX: List[Tuple[str, str]] = [
        ("ICD_CODE",         r"\b[A-Z]\d{2}\.?\d*\b"),
        ("KIDNEY_DISEASE",   r"\b(?:end[\s\-]+stage\s+renal(?:\s+disease)?|esrd|chronic\s+(?:kidney|renal)\s+(?:disease|failure)|renal\s+(?:failure|insufficiency)|kidney\s+failure|ckd(?:\s+stage\s+\d[ab]?)?)\b"),
        ("DIALYSIS",         r"\b(?:hemo|peritoneal\s+)?dialysis\b"),
        ("KIDNEY_FUNCTION",  r"\b(?:e?gfr|glomerular\s+filtration\s+rate?|creatinine|egfr\s+result|gfr\s+result)\b"),
        ("HOSPICE",          r"\bhospice(?:\s+(?:care|enrollment|patient|status))?\b"),
        ("PALLIATIVE",       r"\bpalliative(?:\s+(?:care|status|treatment|intent))?\b"),
        ("COMFORT_CARE",     r"\bcomfort\s+(?:care|measures)(?:\s+only)?\b"),
        ("END_OF_LIFE",      r"\b(?:do\s+not\s+resuscitate|dnr|end[\s\-]+of[\s\-]+life)\b"),
        ("HEART_DISEASE",    r"\b(?:congestive\s+heart\s+failure|chf|myocardial\s+infarction|heart\s+failure|atrial\s+fibrillation)\b"),
        ("LUNG_DISEASE",     r"\b(?:chronic\s+obstructive\s+pulmonary\s+disease|copd|pulmonary\s+embolism|pulmonary\s+fibrosis)\b"),
        ("DIABETES",         r"\b(?:type\s+[12]\s+diabetes(?:\s+mellitus)?|diabetes\s+mellitus|insulin[\-\s]+dependent\s+diabetes|t[12]dm)\b"),
        ("CANCER",           r"\b(?:malignant\s+neoplasm|carcinoma|metastatic\s+cancer|lymphoma|leukemia)\b"),
        ("STROKE",           r"\b(?:cerebrovascular\s+accident|cva|ischemic\s+stroke|hemorrhagic\s+stroke|transient\s+ischemic\s+attack|tia)\b"),
        ("HYPERTENSION",     r"\b(?:hypertens(?:ion|ive)|high\s+blood\s+pressure|htn)\b"),
        ("TRANSPLANT",       r"\b(?:kidney|renal)\s+transplant\b"),
    ]

    # ── Negation cues ────────────────────────────────────────────────────────
    _NEG_WORDS: frozenset = frozenset({
        "no", "not", "without", "denies", "denied",
        "absent", "never", "none", "negative", "unlikely",
    })
    _NEG_PHRASES: Tuple[str, ...] = (
        "ruled out", "rule out", "free of", "no evidence", "no sign",
        "does not have", "doesn't have",
    )
    _NEG_WINDOW: int = 6

    # ── Temporal qualifiers (inline "history of X") ──────────────────────────
    _HISTORICAL_PHRASES: Tuple[str, ...] = (
        "history of", "h/o ", "hx of", "hx:",
        "previous ", "prior ", "former ", "remote history",
        "had a ", "had an ", "years ago", "in the past",
    )

    # ── Section headers ──────────────────────────────────────────────────────
    _SECTION_DEFS: List[Tuple[str, str]] = [
        (r"\b(?:family\s+history|family\s+hx)\s*[:\-]",           "FAMILY_HISTORY"),
        (r"\b(?:past\s+(?:medical\s+)?history|pmh)\s*[:\-]",      "PAST_HISTORY"),
        (r"\b(?:surgical\s+history|psh)\s*[:\-]",                 "PAST_HISTORY"),
        (r"\b(?:social\s+history|social\s+hx)\s*[:\-]",           "SOCIAL_HISTORY"),
        (r"\b(?:assessment|impression|diagnosis|diagnoses)\s*[:\-]", "CURRENT"),
        (r"\b(?:history\s+of\s+present\s+illness|hpi)\s*[:\-]",   "CURRENT"),
        (r"\b(?:chief\s+complaint|cc)\s*[:\-]",                    "CURRENT"),
        (r"\bplan\s*[:\-]",                                         "CURRENT"),
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self.nlp = None

        self._compiled_regex: List[Tuple[str, re.Pattern]] = [
            (label, re.compile(pattern, re.IGNORECASE))
            for label, pattern in self._REGEX
        ]
        self._compiled_sections: List[Tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), label)
            for pattern, label in self._SECTION_DEFS
        ]

        if SPACY_AVAILABLE:
            self._load_spacy()

    def _load_spacy(self) -> None:
        for model in ["en_core_sci_lg", "en_core_sci_md", "en_core_sci_sm",
                      "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
            try:
                self.nlp = spacy.load(model)
                log.info(f"spaCy model: {model}")
                break
            except OSError:
                continue

        if not self.nlp:
            log.warning("No spaCy model found — regex-only extraction.")
            return

        self.nlp.max_length = max(self.nlp.max_length, self._config.max_text_chars + 1_000)

        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns: List[Dict] = []
            for label, terms in self._RULER_ENTRIES:
                for term in terms:
                    patterns.append({"label": label, "pattern": term})
                    patterns.append({"label": label, "pattern": term.lower()})
            ruler.add_patterns(patterns)
            log.info(f"EntityRuler: {len(patterns)} patterns.")

    # ── Negation ─────────────────────────────────────────────────────────────

    def _is_negated(self, text: str, entity_start: int) -> bool:
        pre = text[:entity_start].lower()
        # Clause-aware: reset at last sentence boundary so "not specified, … GFR" isn't negated
        last_boundary = max(pre.rfind("."), pre.rfind(","), pre.rfind(";"), pre.rfind(":"))
        if last_boundary >= 0:
            pre = pre[last_boundary + 1:]
        words = re.findall(r"\b\w+\b", pre)[-self._NEG_WINDOW:]
        window = " ".join(words)
        return bool(self._NEG_WORDS & set(words)) or any(p in window for p in self._NEG_PHRASES)

    # ── Temporal ─────────────────────────────────────────────────────────────

    def _is_historical(self, text: str, entity_start: int) -> bool:
        """True when an inline temporal qualifier precedes the entity in the same clause."""
        pre = text[:entity_start].lower()
        last_boundary = max(pre.rfind("."), pre.rfind(";"))
        if last_boundary >= 0:
            pre = pre[last_boundary + 1:]
        context = pre[-80:]
        return any(phrase in context for phrase in self._HISTORICAL_PHRASES)

    # ── Section context ───────────────────────────────────────────────────────

    def _compute_section_map(self, text: str) -> List[Tuple[int, str]]:
        sections: List[Tuple[int, str]] = []
        for pattern, label in self._compiled_sections:
            for m in pattern.finditer(text):
                sections.append((m.start(), label))
        return sorted(sections)

    @staticmethod
    def _section_at(section_map: List[Tuple[int, str]], pos: int) -> str:
        lo, hi, result = 0, len(section_map) - 1, "CURRENT"
        while lo <= hi:
            mid = (lo + hi) // 2
            if section_map[mid][0] <= pos:
                result = section_map[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # ── CKD stage extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_stage(text: str, start: int, end: int, label: str) -> Optional[str]:
        """Extract CKD stage number from entity context (±50 chars)."""
        if label != "KIDNEY_DISEASE":
            return None
        ctx = text[max(0, start - 30): end + 50].lower()
        m = re.search(r"\bstage\s+(\d+[ab]?)\b", ctx) or re.search(r"\bckd\s*(\d+[ab]?)\b", ctx)
        return m.group(1) if m else None

    # ── eGFR value extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_egfr_value(text: str, end: int, label: str) -> Optional[float]:
        """Extract numeric eGFR from the 40 chars immediately after the entity."""
        if label != "KIDNEY_FUNCTION":
            return None
        ctx = text[end: end + 60]
        m = re.search(r"[\s=:]*(\d+(?:\.\d+)?)", ctx)
        if m:
            val = float(m.group(1))
            # Clinical eGFR range: 5–150 mL/min. Excludes unit-normalisation
            # factors like /1.73 m² (BSA) and small multipliers like 1.21.
            if 5.0 <= val <= 150.0:
                return val
        return None

    # ── Document-level contextual inference ─────────────────────────────────
    # Fires when the whole document is about a clinical context even if no
    # single sentence contains an explicit diagnosis phrase.
    # Each entry: (document_label, [(signal_pattern, weight), ...], threshold)
    _DOC_CONTEXT_SIGNALS: List[Tuple[str, List[Tuple[str, float]], float]] = [
        (
            "KIDNEY_DISEASE",
            [
                (r"\b(?:e?gfr|glomerular\s+filtration)\b",              0.4),
                (r"\b(?:renal\s+failure|chronic\s+renal|ckd|esrd)\b",   0.5),
                (r"\b(?:nephrol|dialysis|creatinine|azotemia)\b",        0.3),
                (r"\b(?:hemodialysis|peritoneal)\b",                     0.5),
                (r"\bstable\s+chronic\b",                                0.2),
                (r"\b(?:kidney|renal)\s+(?:patients?|population)\b",    0.3),
            ],
            0.7,   # cumulative weight threshold to fire
        ),
        (
            "KIDNEY_FUNCTION",
            [
                (r"\b(?:e?gfr)\b",                                       0.5),
                (r"\bmL/min\b",                                          0.3),
                (r"\b1\.73\s*(?:sq\.?\s*m|m\^?2)\b",                    0.3),
                (r"\bglomerular\s+filtration\b",                         0.5),
            ],
            0.5,
        ),
        (
            "HOSPICE",
            [
                (r"\bhospice\b",                                         0.6),
                (r"\bcomfort\s+(?:care|measures)\b",                     0.4),
                (r"\bpalliative\b",                                      0.3),
                (r"\bend[\s\-]+of[\s\-]+life\b",                         0.5),
            ],
            0.6,
        ),
    ]

    def _infer_from_doc_context(self, text: str) -> List[Dict]:
        """
        Adds synthetic entities for labels whose cumulative signal weight
        exceeds the threshold — catches lab-report templates and similar
        context-only documents.
        """
        inferred: List[Dict] = []
        lower = text.lower()
        for label, signals, threshold in self._DOC_CONTEXT_SIGNALS:
            weight = 0.0
            for pattern, w in signals:
                if re.search(pattern, lower):
                    weight += w
                    if weight >= threshold:
                        break
            if weight >= threshold:
                inferred.append({
                    "text":       label.replace("_", " ").lower(),
                    "label":      label,
                    "start":      0,
                    "end":        0,
                    "source":     "doc_context",
                    "negated":    False,
                    "section":    "CURRENT",
                    "temporal":   "CURRENT",
                    "stage":      None,
                    "egfr_value": None,
                    "ctx_weight": round(weight, 2),
                })
        return inferred

    # ── Batch extraction ─────────────────────────────────────────────────────

    def extract_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Process a list of texts in one pass using nlp.pipe() + regex.
        Each entity dict contains:
          text, label, start, end, source,
          negated, section, temporal, stage (KIDNEY_DISEASE), egfr_value (KIDNEY_FUNCTION)
        """
        if not texts:
            return []

        results: List[List[Dict]] = [[] for _ in texts]

        if self.nlp:
            for i, doc in enumerate(self.nlp.pipe(texts, batch_size=self._config.nlp_batch_size)):
                for ent in doc.ents:
                    results[i].append({
                        "text": ent.text, "label": ent.label_,
                        "start": ent.start_char, "end": ent.end_char, "source": "spacy",
                    })

        for i, text in enumerate(texts):
            section_map = self._compute_section_map(text)

            for ent in results[i]:
                ent["negated"]    = self._is_negated(text, ent["start"])
                ent["section"]    = self._section_at(section_map, ent["start"])
                ent["temporal"]   = "HISTORICAL" if self._is_historical(text, ent["start"]) else "CURRENT"
                ent["stage"]      = self._extract_stage(text, ent["start"], ent["end"], ent["label"])
                ent["egfr_value"] = self._extract_egfr_value(text, ent["end"], ent["label"])

            for label, pattern in self._compiled_regex:
                for m in pattern.finditer(text):
                    results[i].append({
                        "text":       text[m.start():m.end()],
                        "label":      label,
                        "start":      m.start(),
                        "end":        m.end(),
                        "source":     "regex",
                        "negated":    self._is_negated(text, m.start()),
                        "section":    self._section_at(section_map, m.start()),
                        "temporal":   "HISTORICAL" if self._is_historical(text, m.start()) else "CURRENT",
                        "stage":      self._extract_stage(text, m.start(), m.end(), label),
                        "egfr_value": self._extract_egfr_value(text, m.end(), label),
                    })

            # Append doc-level inferred entities only when no regex/spaCy
            # entity already covers the same label (avoids duplicates).
            existing_labels = {e["label"] for e in results[i] if not e.get("negated")}
            for inf_ent in self._infer_from_doc_context(text):
                if inf_ent["label"] not in existing_labels:
                    results[i].append(inf_ent)

            results[i] = self._deduplicate(results[i])

        return results

    def extract(self, text: str) -> List[Dict]:
        return self.extract_batch([text])[0]

    @staticmethod
    def _deduplicate(entities: List[Dict]) -> List[Dict]:
        if not entities:
            return entities
        source_rank = {"spacy": 0, "regex": 1}
        sorted_ents = sorted(
            entities,
            key=lambda e: (e["start"], -(e["end"] - e["start"]), source_rank.get(e["source"], 2)),
        )
        result: List[Dict] = []
        last_end = -1
        for ent in sorted_ents:
            if ent["start"] >= last_end:
                result.append(ent)
                last_end = ent["end"]
        return result

# =============================================================================
# STEP 4 — ICD-10 MAPPER  (v3: seeds + label-filtered TF-IDF)
# =============================================================================
class ICD10Mapper:
    """
    Four-stage ICD-10-CM mapping strategy:
      1. Inline ICD code in entity text          (score=1.0, method=direct)
      2. Curated seed table lookup               (score=1.0, method=seed)
      3. Label-filtered TF-IDF cosine similarity (score=variable, method=tfidf-label)
      4. Token-overlap keyword fallback          (score=variable, method=keyword)

    The seed table and label-prefix filtering together ensure that "diabetes
    mellitus" maps to E119 (Type 2 DM) rather than "family history of DM"
    (Z833), and "hospice care" maps to Z515 rather than a generic care code.

    <- ADD HERE (4/4): New seed entries for newly added conditions.
    """

    # ── Curated seed table ───────────────────────────────────────────────────
    # Maps lowercase entity text -> list of ICD-10 codes (no dots).
    # First code in list = primary; all codes in list are returned.
    # <- ADD HERE (4/4): add "entity text": ["ICD_CODE", ...] entries
    _SEEDS: Dict[str, List[str]] = {
        # KIDNEY_DISEASE
        "esrd":                           ["N186"],
        "end stage renal disease":        ["N186"],
        "end-stage renal disease":        ["N186"],
        "ckd stage 5":                    ["N186"],
        "stage 5 ckd":                    ["N186"],
        "ckd stage 4":                    ["N184"],
        "stage 4 ckd":                    ["N184"],
        "ckd stage 3":                    ["N183"],
        "stage 3 ckd":                    ["N183"],
        "ckd stage 2":                    ["N182"],
        "ckd stage 1":                    ["N181"],
        "chronic kidney disease":          ["N189"],
        "chronic renal failure":           ["N189"],
        "chronic renal insufficiency":     ["N189"],
        "renal insufficiency":             ["N189"],
        "ckd":                             ["N189"],
        "renal failure":                   ["N179"],
        "kidney failure":                  ["N179"],
        # DIALYSIS
        "hemodialysis":                   ["Z992", "Z4931"],
        "peritoneal dialysis":            ["Z992", "Z4932"],
        "dialysis":                       ["Z992"],
        "renal dialysis":                 ["Z992"],
        # TRANSPLANT
        "kidney transplant":              ["Z940"],
        "renal transplant":               ["Z940"],
        # HOSPICE / PALLIATIVE / END-OF-LIFE
        "hospice":                        ["Z515"],
        "hospice care":                   ["Z515"],
        "hospice enrollment":             ["Z515"],
        "palliative":                     ["Z515"],
        "palliative care":                ["Z515"],
        "palliative intent":              ["Z515"],
        "comfort care":                   ["Z515"],
        "comfort measures":               ["Z515"],
        "comfort measures only":          ["Z515"],
        "comfort care only":              ["Z515"],
        "do not resuscitate":             ["Z66"],
        "dnr":                            ["Z66"],
        "end of life":                    ["Z515"],
        "terminal illness":               ["Z515"],
        # HEART_DISEASE
        "congestive heart failure":       ["I509"],
        "chf":                            ["I509"],
        "heart failure":                  ["I509"],
        "myocardial infarction":          ["I219"],
        "atrial fibrillation":            ["I489"],
        # LUNG_DISEASE
        "copd":                           ["J449"],
        "chronic obstructive pulmonary disease": ["J449"],
        "pulmonary embolism":             ["I269"],
        "pulmonary fibrosis":             ["J841"],
        # DIABETES
        "type 2 diabetes":                ["E119"],
        "type 2 diabetes mellitus":       ["E119"],
        "t2dm":                           ["E119"],
        "diabetes mellitus":              ["E119"],
        "type 1 diabetes":                ["E109"],
        "type 1 diabetes mellitus":       ["E109"],
        "t1dm":                           ["E109"],
        "insulin dependent diabetes":     ["E109"],
        "insulin-dependent diabetes":     ["E109"],
        # HYPERTENSION
        "hypertension":                   ["I10"],
        "hypertensive":                   ["I10"],
        "high blood pressure":            ["I10"],
        "htn":                            ["I10"],
        # STROKE
        "ischemic stroke":                ["I639"],
        "cerebrovascular accident":       ["I639"],
        "cva":                            ["I639"],
        "hemorrhagic stroke":             ["I619"],
        "transient ischemic attack":      ["G459"],
        "tia":                            ["G459"],
        # CANCER (unspecified — callers with more specific entity text will get better seeds)
        "malignant neoplasm":             ["C809"],
        "metastatic cancer":              ["C800"],
        "carcinoma":                      ["C809"],
        "lymphoma":                       ["C859"],
        "leukemia":                       ["C959"],
    }

    # ── CKD stage number -> ICD code ─────────────────────────────────────────
    _CKD_STAGE_CODES: Dict[str, str] = {
        "1": "N181", "2": "N182",
        "3": "N183", "3a": "N183", "3b": "N183",
        "4": "N184", "5": "N185",
    }

    # ── eGFR ranges -> CKD stage code ────────────────────────────────────────
    # (min_value_inclusive, icd_code) — evaluated in order; first match wins
    _EGFR_RANGES: List[Tuple[float, str]] = [
        (90.0, "N181"),   # >= 90: Stage 1
        (60.0, "N182"),   # 60–89: Stage 2
        (30.0, "N183"),   # 30–59: Stage 3
        (15.0, "N184"),   # 15–29: Stage 4
        (0.0,  "N185"),   # < 15:  Stage 5
    ]

    # ── Label -> ICD chapter prefixes for targeted TF-IDF/keyword search ─────
    _LABEL_PREFIXES: Dict[str, List[str]] = {
        "KIDNEY_DISEASE":  ["N17", "N18", "N19"],
        "DIALYSIS":        ["Z49", "Z99", "R880"],
        "TRANSPLANT":      ["Z94", "Z48"],
        "KIDNEY_FUNCTION": ["N18", "R94", "R798"],
        "HEART_DISEASE":   ["I50", "I21", "I22", "I25", "I48"],
        "LUNG_DISEASE":    ["J44", "J45", "J84", "J96", "J98", "I26"],
        "DIABETES":        ["E10", "E11", "E13"],
        "CANCER":          ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "D0"],
        "STROKE":          ["I63", "I61", "I62", "G45"],
        "HYPERTENSION":    ["I10", "I11", "I12", "I13", "I15"],
        "HOSPICE":         ["Z51"],
        "PALLIATIVE":      ["Z51"],
        "COMFORT_CARE":    ["Z51"],
        "END_OF_LIFE":     ["Z51", "Z66"],
    }

    def __init__(self, icd10_codes: Dict[str, str], config: PipelineConfig) -> None:
        self.codes = icd10_codes
        self._config = config
        self._code_list: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._label_indices: Dict[str, List[int]] = {}

        # Validate seeds — silently drop any seed codes not in the loaded dict
        self._valid_seeds: Dict[str, List[str]] = {}
        for text, code_list in self._SEEDS.items():
            valid = [c for c in code_list if c in icd10_codes]
            if valid:
                self._valid_seeds[text] = valid
            else:
                log.debug(f"Seed '{text}' -> {code_list}: none found in code dict — skipped.")

        log.info(f"Seed table: {len(self._valid_seeds)}/{len(self._SEEDS)} entries validated.")

        if icd10_codes:
            self._build_index()

    def _build_index(self) -> None:
        if not SKLEARN_AVAILABLE:
            log.warning(
                "scikit-learn not installed — keyword fallback only "
                "(Install: pip install scikit-learn numpy)"
            )
            return

        self._code_list = list(self.codes.keys())
        descriptions = [self.codes[c] for c in self._code_list]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, self._config.tfidf_ngram_max),
            lowercase=True,
            max_features=self._config.tfidf_max_features,
        )
        self._matrix = self._vectorizer.fit_transform(descriptions)

        # Pre-compute per-label index subsets for fast targeted search
        for label, prefixes in self._LABEL_PREFIXES.items():
            self._label_indices[label] = [
                i for i, code in enumerate(self._code_list)
                if any(code.startswith(p) for p in prefixes)
            ]
        subset_summary = ", ".join(f"{l}:{len(v)}" for l, v in self._label_indices.items())
        log.info(f"TF-IDF index: {len(self._code_list):,} codes. Label subsets: {subset_summary}")

    # ── Public API ────────────────────────────────────────────────────────────

    def map_entity(self, entity: Dict, top_n: Optional[int] = None) -> List[Dict]:
        """
        Map an entity dict to ICD-10 codes.
        Uses entity['label'] for targeted search and entity['stage'] / ['egfr_value']
        for CKD-specific inference.
        """
        top_n = top_n if top_n is not None else self._config.icd_top_n
        entity_text  = entity.get("text", "")
        entity_label = entity.get("label", "")
        stage        = entity.get("stage")
        egfr_value   = entity.get("egfr_value")

        # Stage 1: inline ICD code in entity text
        direct = self._direct_match(entity_text)
        if direct:
            return direct[:top_n]

        # Stage 2: CKD stage inference from extracted stage number
        if stage and entity_label == "KIDNEY_DISEASE":
            stage_result = self._stage_match(stage)
            if stage_result:
                return [stage_result]

        # Stage 3: eGFR-to-stage inference
        if egfr_value is not None and entity_label == "KIDNEY_FUNCTION":
            egfr_result = self._egfr_match(egfr_value)
            if egfr_result:
                return [egfr_result]

        # Stage 4: curated seed table
        seed = self._seed_match(entity_text)
        if seed:
            return seed[:top_n]

        # Stage 5: TF-IDF within label-relevant code subset
        if SKLEARN_AVAILABLE and self._vectorizer is not None:
            return self._tfidf_match(entity_text, entity_label, top_n)

        # Stage 6: keyword overlap within label-relevant codes (or all codes)
        return self._keyword_match(entity_text, entity_label, top_n)

    # ── Matching strategies ───────────────────────────────────────────────────

    def _direct_match(self, text: str) -> List[Dict]:
        results = []
        for raw in re.findall(r"\b([A-Z]\d{2}\.?\d*)\b", text.upper()):
            key = raw.replace(".", "")
            if key in self.codes:
                results.append({"code": key, "description": self.codes[key], "score": 1.0, "method": "direct"})
        return results

    def _stage_match(self, stage: str) -> Optional[Dict]:
        code = self._CKD_STAGE_CODES.get(stage.lower())
        if code and code in self.codes:
            return {"code": code, "description": self.codes[code], "score": 1.0, "method": "stage"}
        return None

    def _egfr_match(self, egfr_value: float) -> Optional[Dict]:
        code = None
        for threshold, stage_code in self._EGFR_RANGES:
            if egfr_value >= threshold:
                code = stage_code
                break
        if code and code in self.codes:
            return {"code": code, "description": self.codes[code], "score": 0.95, "method": "egfr_inference"}
        return None

    def _seed_match(self, text: str) -> List[Dict]:
        codes = self._valid_seeds.get(text.lower().strip(), [])
        return [
            {"code": c, "description": self.codes[c], "score": 1.0, "method": "seed"}
            for c in codes if c in self.codes
        ]

    def _tfidf_match(self, text: str, label: str, top_n: int) -> List[Dict]:
        try:
            vec = self._vectorizer.transform([text])
            label_idx = self._label_indices.get(label)

            if label_idx:
                # Search within label-relevant code subset first
                sub = self._matrix[label_idx]
                sims = cosine_similarity(vec, sub)[0]
                top_j = np.argsort(sims)[::-1][:top_n]
                results = [
                    {
                        "code":        self._code_list[label_idx[j]],
                        "description": self.codes[self._code_list[label_idx[j]]],
                        "score":       round(float(sims[j]), 4),
                        "method":      "tfidf-label",
                    }
                    for j in top_j if sims[j] > self._config.tfidf_min_score
                ]
                if results:
                    return results

            # Fallback: full TF-IDF
            sims_full = cosine_similarity(vec, self._matrix)[0]
            top_i = np.argsort(sims_full)[::-1][:top_n]
            return [
                {
                    "code":        self._code_list[i],
                    "description": self.codes[self._code_list[i]],
                    "score":       round(float(sims_full[i]), 4),
                    "method":      "tfidf",
                }
                for i in top_i if sims_full[i] > self._config.tfidf_min_score
            ]
        except Exception as exc:
            log.warning(f"TF-IDF error: {exc}")
            return self._keyword_match(text, label, top_n)

    def _keyword_match(self, text: str, label: str, top_n: int) -> List[Dict]:
        query = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        if not query:
            return []

        prefixes = self._LABEL_PREFIXES.get(label, [])
        scored = []

        def _score(code_desc_pairs):
            for code, desc in code_desc_pairs:
                overlap = len(query & set(re.findall(r"\b[a-z]{3,}\b", desc.lower())))
                if overlap:
                    scored.append((overlap / max(len(query), 1), code, desc))

        if prefixes:
            _score(
                (c, d) for c, d in self.codes.items()
                if any(c.startswith(p) for p in prefixes)
            )

        if not scored:    # label filter yielded nothing — try all codes
            _score(self.codes.items())

        scored.sort(reverse=True)
        return [
            {"code": c, "description": d, "score": round(s, 4), "method": "keyword"}
            for s, c, d in scored[:top_n]
        ]


# =============================================================================
# STEP 5 — PIPELINE ORCHESTRATOR
# =============================================================================
class EnhancedTextMiningPipeline:
    """
    Orchestrates all pipeline stages.
    v3 adds temporal filtering and passes full entity dicts (with stage/eGFR)
    to the mapper so stage-specific and eGFR-inferred codes are used.
    """

    _EXPECTED_FIELDS: frozenset = frozenset(
        {"DOCUMENT_ID", "CONTACT_LINE", "CONTACT_DATE", "RESULT_TEXT"}
    )

    # Only these labels are eligible for ICD-10 mapping.
    # Excludes spaCy general NER labels (CARDINAL, DATE, NORP, ORG, PERSON …)
    # which produce noise when passed through TF-IDF against 98k medical codes.
    _MEDICAL_LABELS: frozenset = frozenset({
        "KIDNEY_DISEASE", "DIALYSIS", "TRANSPLANT", "KIDNEY_FUNCTION",
        "HOSPICE", "PALLIATIVE", "COMFORT_CARE", "END_OF_LIFE",
        "HEART_DISEASE", "LUNG_DISEASE", "DIABETES", "CANCER",
        "STROKE", "HYPERTENSION", "ICD_CODE",
    })

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        log.info("=" * 65)
        log.info("  Enhanced KP Text Mining Pipeline v3 — Initializing")
        log.info("=" * 65)

        icd10_codes = ICD10CMFetcher(force_refresh=self.config.force_icd_refresh).fetch()
        self.spell_checker    = ClinicalSpellChecker(self.config)
        self.entity_extractor = MedicalEntityExtractor(self.config)
        self.icd10_mapper     = ICD10Mapper(icd10_codes, self.config)

        log.info("Pipeline ready.\n")

    def _validate_records(self, records: List[Dict]) -> None:
        if not records:
            log.warning("run() called with empty record list.")
            return
        missing = self._EXPECTED_FIELDS - records[0].keys()
        if missing:
            log.warning(
                f"Records missing expected fields: {sorted(missing)}. "
                "Expected: DOCUMENT_ID, CONTACT_LINE, CONTACT_DATE, RESULT_TEXT."
            )

    def run(self, records: List[Dict], log_every: Optional[int] = None) -> List[Dict]:
        """
        Process a batch; return only records with at least one flagged entity.

        v3 processing flow:
          Phase 1 — spell-correct all records
          Phase 2 — batch spaCy NER via nlp.pipe() (single vectorised pass)
          Phase 3 — ICD mapping with seed + label-filtered TF-IDF, optionally parallel
        """
        log_every = log_every if log_every is not None else self.config.batch_log_every
        total = len(records)
        log.info(f"Processing {total:,} record(s) …")
        self._validate_records(records)

        # ── Phase 1: spell correction ─────────────────────────────────────────
        prepared: List[Optional[Tuple]] = []
        for i, rec in enumerate(records, 1):
            try:
                doc_id   = f"{rec.get('DOCUMENT_ID', 'UNKNOWN')}_{rec.get('CONTACT_LINE', 0)}"
                raw_text = str(rec.get("RESULT_TEXT", "") or "").strip()
                if len(raw_text) > self.config.max_text_chars:
                    log.warning(f"[{doc_id}] truncated to {self.config.max_text_chars:,} chars.")
                    raw_text = raw_text[:self.config.max_text_chars]
                corrected, fixes = self.spell_checker.correct(raw_text)
                prepared.append((rec, corrected, fixes, doc_id, raw_text))
            except Exception as exc:
                log.error(f"Record #{i} prep failed: {exc} — skipped.")
                prepared.append(None)

        # ── Phase 2: batch NER ────────────────────────────────────────────────
        valid_idx   = [i for i, p in enumerate(prepared) if p is not None]
        valid_texts = [prepared[i][1] for i in valid_idx]

        raw_entities: Dict[int, List[Dict]] = {}
        if valid_texts:
            try:
                batch_ents = self.entity_extractor.extract_batch(valid_texts)
                raw_entities = {idx: ents for idx, ents in zip(valid_idx, batch_ents)}
            except Exception as exc:
                log.error(f"Batch NER failed ({exc}) — falling back to per-record.")
                for j, idx in enumerate(valid_idx):
                    try:
                        raw_entities[idx] = self.entity_extractor.extract(valid_texts[j])
                    except Exception as e2:
                        log.error(f"Per-record NER failed for #{idx+1}: {e2}")
                        raw_entities[idx] = []

        # ── Phase 3: ICD mapping + assembly ──────────────────────────────────
        def _assemble(idx: int) -> Optional[Dict]:
            prep = prepared[idx]
            if prep is None:
                return None
            rec, corrected, fixes, doc_id, raw_text = prep

            all_ents = raw_entities.get(idx, [])

            # Apply accuracy filters
            entities = [
                e for e in all_ents
                if not (self.config.exclude_negated   and e.get("negated", False))
                and e.get("section", "CURRENT") not in self.config.exclude_sections
                and not (self.config.exclude_historical and e.get("temporal") == "HISTORICAL")
            ]

            icd_mappings: List[Dict] = []
            seen: set = set()
            for ent in entities:
                if ent["label"] not in self._MEDICAL_LABELS:
                    continue  # skip spaCy general-NER entities (CARDINAL, DATE, NORP…)
                for match in self.icd10_mapper.map_entity(ent, self.config.icd_top_n):
                    key = (match["code"], ent["text"][:50])
                    if key not in seen:
                        seen.add(key)
                        icd_mappings.append({
                            "entity_text":  ent["text"],
                            "entity_label": ent["label"],
                            "stage":        ent.get("stage"),
                            "egfr_value":   ent.get("egfr_value"),
                            **match,
                        })

            return {
                "doc_id":            doc_id,
                "document_id":       rec.get("DOCUMENT_ID"),
                "contact_line":      rec.get("CONTACT_LINE"),
                "contact_date":      str(rec.get("CONTACT_DATE", "")),
                "raw_text":          raw_text,
                "corrected_text":    corrected,
                "spell_corrections": fixes,
                "entities":          entities,
                "all_entities":      all_ents,
                "icd10_mappings":    icd_mappings,
                "text_snippet":      raw_text[:250],
            }

        results: List[Dict] = []
        width = len(str(total))

        if self.config.worker_threads > 1:
            ordered: List[Optional[Dict]] = [None] * total
            with ThreadPoolExecutor(max_workers=self.config.worker_threads) as pool:
                future_map = {pool.submit(_assemble, i): i for i in range(total)}
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        ordered[idx] = fut.result()
                    except Exception as exc:
                        log.error(f"Worker error on record #{idx+1}: {exc}")
            results = [r for r in ordered if r and (r["entities"] or r["icd10_mappings"])]
        else:
            for i in range(total):
                try:
                    result = _assemble(i)
                except Exception as exc:
                    log.error(f"Record #{i+1} (Doc {records[i].get('DOCUMENT_ID','?')}) failed: {exc}")
                    continue

                if result is None:
                    continue

                flagged = bool(result["entities"] or result["icd10_mappings"])
                if flagged:
                    results.append(result)

                if flagged or ((i + 1) % log_every == 0) or (i + 1 == total):
                    all_e = result.get("all_entities", result["entities"])
                    neg_e  = [e for e in all_e if e.get("negated")]
                    hist_e = [e for e in all_e if e.get("temporal") == "HISTORICAL"]
                    log.info(
                        f"  [{i+1:>{width}}/{total}]"
                        f" Doc {records[i].get('DOCUMENT_ID','?')}"
                        f" | raw={len(all_e)} neg={len(neg_e)} hist={len(hist_e)}"
                        f" | kept={len(result['entities'])}"
                        f" | icd10={len(result['icd10_mappings'])}"
                        f" | fixes={len(result['spell_corrections'])}"
                        f" | {'YES <<' if flagged else 'no'}"
                    )

        log.info(f"\nFlagged: {len(results):,} / {total:,} records\n")
        return results


# =============================================================================
# STEP 6 — EXPORT  (CSV + SQL + JSONL)
# =============================================================================

def export_csv(results: List[Dict], filepath: str = "enhanced_results.csv", append: bool = False) -> None:
    rows: List[Dict] = []
    for r in results:
        base = {
            "DOCUMENT_ID":  r["document_id"],
            "CONTACT_LINE": r["contact_line"],
            "CONTACT_DATE": r["contact_date"],
            "TEXT_SNIPPET": r["text_snippet"][:200],
            "SPELL_FIXES":  "; ".join(f"{c['original']}->{c['corrected']}" for c in r["spell_corrections"]),
            "PROCESSED_TS": datetime.now().isoformat(),
        }
        if r["icd10_mappings"]:
            for m in r["icd10_mappings"]:
                rows.append({
                    **base,
                    "ENTITY_TEXT":       m["entity_text"],
                    "ENTITY_LABEL":      m["entity_label"],
                    "STAGE":             m.get("stage") or "",
                    "EGFR_VALUE":        m.get("egfr_value") or "",
                    "ICD10_CODE":        m["code"],
                    "ICD10_DESCRIPTION": m["description"],
                    "MATCH_SCORE":       m["score"],
                    "MATCH_METHOD":      m["method"],
                })
        else:
            for ent in r["entities"]:
                rows.append({
                    **base,
                    "ENTITY_TEXT":       ent["text"],
                    "ENTITY_LABEL":      ent["label"],
                    "STAGE":             ent.get("stage") or "",
                    "EGFR_VALUE":        ent.get("egfr_value") or "",
                    "ICD10_CODE":        "",
                    "ICD10_DESCRIPTION": "",
                    "MATCH_SCORE":       0,
                    "MATCH_METHOD":      "none",
                })

    if not rows:
        log.warning("[EXPORT] CSV — no rows to write.")
        return

    mode = "a" if append else "w"
    with open(filepath, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    log.info(f"[EXPORT] CSV  -> {filepath}  ({len(rows)} rows, append={append})")


def export_sql_inserts(results: List[Dict], filepath: str = "enhanced_results.sql", append: bool = False) -> None:
    lines: List[str] = []
    if not append:
        lines = [
            "-- Auto-generated by Enhanced KP Text Mining Pipeline v3",
            f"-- Generated: {datetime.now().isoformat()}",
            "",
            "CREATE TABLE IF NOT EXISTS clinical_icd10_flags (",
            "    DOCUMENT_ID       BIGINT       NOT NULL,",
            "    CONTACT_LINE      INT,",
            "    CONTACT_DATE      VARCHAR(30),",
            "    ENTITY_TEXT       VARCHAR(500),",
            "    ENTITY_LABEL      VARCHAR(100),",
            "    STAGE             VARCHAR(10),",
            "    EGFR_VALUE        DECIMAL(6,1),",
            "    ICD10_CODE        VARCHAR(20),",
            "    ICD10_DESCRIPTION VARCHAR(500),",
            "    MATCH_SCORE       DECIMAL(8,4),",
            "    MATCH_METHOD      VARCHAR(50),",
            "    SPELL_FIXES       VARCHAR(500),",
            "    TEXT_SNIPPET      VARCHAR(300),",
            "    PROCESSED_TS      TIMESTAMP",
            ");",
            "",
        ]

    for r in results:
        snippet = r["text_snippet"][:290].replace("'", "''")
        fixes   = "; ".join(f"{c['original']}->{c['corrected']}" for c in r["spell_corrections"]).replace("'", "''")[:490]
        cl      = r["contact_line"] if r["contact_line"] is not None else "NULL"
        mappings = r["icd10_mappings"] or [
            {"entity_text": e["text"], "entity_label": e["label"],
             "stage": e.get("stage"), "egfr_value": e.get("egfr_value"),
             "code": "", "description": "", "score": 0, "method": "none"}
            for e in r["entities"]
        ]
        for m in mappings:
            entity = m["entity_text"].replace("'", "''")[:490]
            desc   = m["description"].replace("'", "''")[:490]
            stage  = f"'{m.get('stage') or ''}'"
            egfr   = str(m.get("egfr_value") or "NULL")
            lines.append(
                f"INSERT INTO clinical_icd10_flags VALUES ("
                f"{r['document_id']}, {cl}, '{r['contact_date']}', "
                f"'{entity}', '{m['entity_label']}', {stage}, {egfr}, "
                f"'{m['code']}', '{desc}', {m['score']}, '{m['method']}', "
                f"'{fixes}', '{snippet}', CURRENT_TIMESTAMP);"
            )

    mode = "a" if append else "w"
    with open(filepath, mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info(f"[EXPORT] SQL  -> {filepath}  (append={append})")


def export_json(results: List[Dict], filepath: str = "enhanced_results.jsonl", append: bool = False) -> None:
    """JSONL: one record per line; includes all_entities, stage, egfr_value, temporal for full audit."""
    mode = "a" if append else "w"
    with open(filepath, mode, encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, default=str) + "\n")
    log.info(f"[EXPORT] JSONL -> {filepath}  ({len(results)} records, append={append})")


# =============================================================================
# STEP 7 — CONSOLE SUMMARY
# =============================================================================

def print_summary(results: List[Dict]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  ENHANCED NLP PIPELINE v3 — DETECTION SUMMARY")
    print(sep)

    if not results:
        print("  No conditions flagged.")
        print(sep)
        return

    for r in results:
        print(f"\n  Doc {r['document_id']}  |  Line {r['contact_line']}  |  {r['contact_date']}")
        print(f"  Text   : {r['text_snippet'][:100]} …")

        if r["spell_corrections"]:
            print(f"  Fixes  : {', '.join(c['original']+'->'+c['corrected'] for c in r['spell_corrections'])}")

        if r["entities"]:
            labels = sorted({e["label"] for e in r["entities"]})
            print(f"  Kept   ({len(r['entities'])}): {', '.join(labels)}")

        all_ents  = r.get("all_entities", r["entities"])
        negated   = [e for e in all_ents if e.get("negated")]
        historical = [e for e in all_ents if e.get("temporal") == "HISTORICAL"]
        sect_excl  = [e for e in all_ents if e.get("section") in ("FAMILY_HISTORY", "PAST_HISTORY")]
        if negated:
            print(f"  Negated  ({len(negated)} excl.): {', '.join({e['text'] for e in negated})}")
        if historical:
            print(f"  Hist.    ({len(historical)} excl.): {', '.join({e['text'] for e in historical})}")
        if sect_excl:
            print(f"  Section  ({len(sect_excl)} excl.): {', '.join({e['text'] for e in sect_excl})}")

        if r["icd10_mappings"]:
            print(f"  ICD-10 ({len(r['icd10_mappings'])}):")
            for m in r["icd10_mappings"][:6]:
                stage_str = f" [Stage {m['stage']}]" if m.get("stage") else ""
                egfr_str  = f" [eGFR={m['egfr_value']}]" if m.get("egfr_value") else ""
                print(
                    f"    >> {m['code']:<8} {m['description'][:52]:<52}"
                    f"  {m['score']:.3f} [{m['method']}]"
                    f"{stage_str}{egfr_str}"
                    f"  <- \"{m['entity_text']}\""
                )

    print(f"\n{sep}")
    print(f"  Total flagged: {len(results)} record(s)")
    print(f"{sep}\n")


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enhanced KP Text Mining Pipeline v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--auto-install",          action="store_true",
                   help="Auto-install missing optional packages and restart")
    p.add_argument("--force-refresh",         action="store_true",
                   help="Re-download ICD-10-CM codes ignoring cache")
    p.add_argument("--top-n",                 type=int,   default=3)
    p.add_argument("--nlp-batch-size",        type=int,   default=64)
    p.add_argument("--workers",               type=int,   default=1)
    p.add_argument("--output-dir",            default=".")
    p.add_argument("--include-negated",       action="store_true",
                   help="Include negated entities in output")
    p.add_argument("--include-family-history",action="store_true",
                   help="Include FAMILY_HISTORY section entities")
    p.add_argument("--exclude-historical",    action="store_true",
                   help="Exclude inline 'history of X' entities")
    p.add_argument("--tfidf-min-score",       type=float, default=0.05)
    p.add_argument("--log-every",             type=int,   default=100)
    return p.parse_args()


# =============================================================================
# SAMPLE RECORDS  (Option A — covers all intelligence features)
# =============================================================================
SAMPLE_RECORDS: List[Dict] = [
    # ── Original records ─────────────────────────────────────────────────────
    {
        "DOCUMENT_ID": 1708275951, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "GFR results reported in mL/min/1.73 sq.m. "
            "For African American individuals where race was not specified, "
            "please multiply the GFR by 1.21. "
            "This eGFR is validated for stable chronic renal failure patients."
        ),
    },
    {
        "DOCUMENT_ID": 1708275951, "CONTACT_LINE": 2, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": "Normal renal function noted. No dialysis required.",
    },
    {
        "DOCUMENT_ID": 9999000001, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Patient has been enrolled in hospice care per family request. "
            "Comfort care measures only. Z51.5 documented."
        ),
    },
    {
        "DOCUMENT_ID": 9999000002, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": "ESRD patient on hemodialysis three times weekly. Kidney transplant workup deferred. ICD N18.6.",
    },
    {
        "DOCUMENT_ID": 9999000003, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Patient diagnosed with congestive heart failure and COPD. "
            "Type 2 diabetes mellitus with nephropathy. "
            "Palliative intent discussed with patient and family."
        ),
    },
    {
        "DOCUMENT_ID": 9999000004, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "The pateint has been refered to nephrolgy for evaluation of "
            "cronic kidny disease stage 4. eGFR 22 mL/min. "
            "Dialysis planninng to begin next quarter."
        ),
    },
    # ── Tests for negation (should NOT be flagged) ────────────────────────────
    {
        "DOCUMENT_ID": 9999000005, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Patient denies chest pain. No congestive heart failure noted. "
            "Without hypertension at this visit."
        ),
    },
    # ── Tests for section filtering (should NOT be flagged) ──────────────────
    {
        "DOCUMENT_ID": 9999000006, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Family History: Hypertension and diabetes mellitus in mother. ESRD in father. "
            "Assessment: Patient currently has no renal disease."
        ),
    },
    # ── NEW: CKD stage detection ──────────────────────────────────────────────
    {
        "DOCUMENT_ID": 9999000007, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Patient has CKD stage 3b with progressive decline. "
            "Nephrology follow-up scheduled. No dialysis at this time."
        ),
    },
    # ── NEW: eGFR-to-stage inference ─────────────────────────────────────────
    {
        "DOCUMENT_ID": 9999000008, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": "eGFR = 18 mL/min/1.73m2. Consistent with Stage 4 CKD. Referral for dialysis planning.",
    },
    # ── NEW: inline temporal qualifier ───────────────────────────────────────
    {
        "DOCUMENT_ID": 9999000009, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Patient has a history of myocardial infarction in 2019. "
            "Currently presents with new onset hypertension and COPD."
        ),
    },
    # ── NEW: multi-condition + specific ICD ──────────────────────────────────
    {
        "DOCUMENT_ID": 9999000010, "CONTACT_LINE": 1, "CONTACT_DATE": "2026-02-10",
        "RESULT_TEXT": (
            "Hypertensive CKD patient (Stage 4, eGFR 24) with Type 2 diabetes mellitus. "
            "On hemodialysis three times weekly. DNR order in place."
        ),
    },
]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = _parse_args()

    # Auto-install missing packages and restart if requested
    if args.auto_install:
        _auto_install_and_restart()

    exclude_sections = ["FAMILY_HISTORY", "PAST_HISTORY", "SOCIAL_HISTORY"]
    if args.include_family_history:
        exclude_sections = ["PAST_HISTORY", "SOCIAL_HISTORY"]

    config = PipelineConfig(
        force_icd_refresh  = args.force_refresh,
        icd_top_n          = args.top_n,
        nlp_batch_size     = args.nlp_batch_size,
        worker_threads     = args.workers,
        output_dir         = args.output_dir,
        exclude_negated    = not args.include_negated,
        exclude_sections   = exclude_sections,
        exclude_historical = args.exclude_historical,
        tfidf_min_score    = args.tfidf_min_score,
        batch_log_every    = args.log_every,
    )

    pipeline = EnhancedTextMiningPipeline(config)

    # =========================================================================
    # OPTION A — Sample records
    # =========================================================================
    out = Path(config.output_dir)
    results = pipeline.run(SAMPLE_RECORDS)

    print_summary(results)

    export_csv(results,         filepath=str(out / "enhanced_results.csv"))
    export_sql_inserts(results, filepath=str(out / "enhanced_results.sql"))
    export_json(results,        filepath=str(out / "enhanced_results.jsonl"))

    log.info("All exports complete.")

    # =========================================================================
    # OPTION B — Databricks  (activate by setting RUN_DATABRICKS=1 env var)
    # Set DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP env vars first.
    # =========================================================================
    if os.getenv("RUN_DATABRICKS", "0") == "1":
        if not DATABRICKS_AVAILABLE:
            log.error("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
            sys.exit(1)

        ckpt            = CheckpointManager(config.checkpoint_file)
        state           = ckpt.load()
        last_anchor     = state["anchor_value"] if state else None
        total_processed = state["total_processed"] if state else 0
        total_flagged   = 0
        batch_number    = state["batch_num"] if state else 0

        QUERY = (
            f"SELECT {ANCHOR_COL} AS DOCUMENT_ID, CONTACT_LINE, CONTACT_DATE, "
            f"{TEXT_COLUMN} AS RESULT_TEXT "
            f"FROM {TABLE_NAME} "
            f"WHERE {TEXT_COLUMN} IS NOT NULL"
        )

        log.info(f"[Databricks] Starting full extraction from {TABLE_NAME} …")
        try:
            for batch in fetch_from_databricks(QUERY, last_anchor=last_anchor):
                batch_number    += 1
                batch_results    = pipeline.run(batch, log_every=config.batch_log_every)
                total_processed += len(batch)
                total_flagged   += len(batch_results)
                is_first         = batch_number == 1 and last_anchor is None

                if batch_results:
                    export_csv(
                        batch_results,
                        filepath=str(out / "enhanced_results.csv"),
                        append=not is_first,
                    )
                    export_sql_inserts(
                        batch_results,
                        filepath=str(out / "enhanced_results.sql"),
                        append=not is_first,
                    )
                    export_json(
                        batch_results,
                        filepath=str(out / "enhanced_results.jsonl"),
                        append=not is_first,
                    )

                ckpt.save(batch[-1][ANCHOR_COL], batch_number, total_processed)

                log.info(
                    f"[Databricks] Batch {batch_number:,}  "
                    f"processed={total_processed:,}  flagged={total_flagged:,}"
                )
        except KeyboardInterrupt:
            log.warning("[Databricks] Interrupted — checkpoint saved. Re-run to resume.")
        except Exception as exc:
            log.error(f"[Databricks] Fatal: {exc}")
            raise
        else:
            ckpt.clear()
            log.info(
                f"[Databricks] Complete. "
                f"Processed: {total_processed:,}  Flagged: {total_flagged:,}"
            )
