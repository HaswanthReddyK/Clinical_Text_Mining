"""
Ground-truth accuracy evaluation for the Enhanced KP Text Mining Pipeline.
Measures Precision, Recall, F1 at both record level and label level.
"""
import sys
sys.path.insert(0, ".")
from enhanced_text_mining_pipeline import EnhancedTextMiningPipeline, PipelineConfig

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH  — 30 records covering every condition, abbreviation,
#                 negation, section, and clinical phrasing variant
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    # ── POSITIVE: clear clinical text ────────────────────────────────────────
    {
        "id": 1,  "note": "ESRD + hemodialysis",
        "text": "Patient has ESRD on hemodialysis three times weekly.",
        "expected": {"KIDNEY_DISEASE", "DIALYSIS"}, "flag": True,
    },
    {
        "id": 2,  "note": "CHF + type 2 diabetes",
        "text": "Diagnosed with congestive heart failure and type 2 diabetes mellitus.",
        "expected": {"HEART_DISEASE", "DIABETES"}, "flag": True,
    },
    {
        "id": 3,  "note": "Hospice + DNR",
        "text": "Patient enrolled in hospice care. DNR order documented.",
        "expected": {"HOSPICE", "END_OF_LIFE"}, "flag": True,
    },
    {
        "id": 4,  "note": "CKD stage 3 + eGFR value",
        "text": "Chronic kidney disease stage 3. eGFR 42 mL/min.",
        "expected": {"KIDNEY_DISEASE", "KIDNEY_FUNCTION"}, "flag": True,
    },
    {
        "id": 5,  "note": "HTN + COPD",
        "text": "Hypertension well controlled. COPD with recent exacerbation.",
        "expected": {"HYPERTENSION", "LUNG_DISEASE"}, "flag": True,
    },
    {
        "id": 6,  "note": "Kidney transplant",
        "text": "Kidney transplant status. Post-op follow-up at 6 weeks.",
        "expected": {"TRANSPLANT"}, "flag": True,
    },
    {
        "id": 7,  "note": "Comfort care + palliative",
        "text": "Comfort measures only. Palliative care consult placed.",
        "expected": {"COMFORT_CARE", "PALLIATIVE"}, "flag": True,
    },
    {
        "id": 8,  "note": "Ischemic stroke",
        "text": "Ischemic stroke confirmed on MRI. CVA workup complete.",
        "expected": {"STROKE"}, "flag": True,
    },
    {
        "id": 9,  "note": "Cancer (metastatic)",
        "text": "Metastatic carcinoma of the lung. Lymphoma ruled in on biopsy.",
        "expected": {"CANCER"}, "flag": True,
    },
    {
        "id": 10, "note": "End-of-life / terminal illness",
        "text": "Patient is end-of-life. Terminal illness discussion documented.",
        "expected": {"END_OF_LIFE"}, "flag": True,
    },
    # ── ABBREVIATION variants ─────────────────────────────────────────────────
    {
        "id": 11, "note": "All abbreviations: CKD HTN CHF COPD",
        "text": "CKD stage 4. HTN. CHF. COPD.",
        "expected": {"KIDNEY_DISEASE", "HYPERTENSION", "HEART_DISEASE", "LUNG_DISEASE"},
        "flag": True,
    },
    {
        "id": 12, "note": "eGFR + HD (hemodialysis abbreviation)",
        "text": "eGFR 18 ml/min. Patient requires HD three times weekly.",
        "expected": {"KIDNEY_FUNCTION", "DIALYSIS"}, "flag": True,
    },
    {
        "id": 13, "note": "Peritoneal dialysis written out",
        "text": "Patient on peritoneal dialysis at home.",
        "expected": {"DIALYSIS"}, "flag": True,
    },
    {
        "id": 14, "note": "Inline ICD codes N18.6 and I10",
        "text": "Primary diagnosis: N18.6. Secondary: I10.",
        "expected": {"ICD_CODE"}, "flag": True,
    },
    {
        "id": 15, "note": "GFR + creatinine (kidney function markers)",
        "text": "Glomerular filtration rate declining. Creatinine elevated at 4.2.",
        "expected": {"KIDNEY_FUNCTION"}, "flag": True,
    },
    # ── NEGATION: conditions denied — should NOT be flagged ──────────────────
    {
        "id": 16, "note": "[NEG] No heart failure, without hypertension",
        "text": "No evidence of heart failure. Without hypertension at this visit.",
        "expected": set(), "flag": False,
    },
    {
        "id": 17, "note": "[NEG] Denies dialysis, no CKD",
        "text": "Patient denies dialysis. No CKD documented.",
        "expected": set(), "flag": False,
    },
    {
        "id": 18, "note": "[NEG] Ruled out COPD and diabetes",
        "text": "COPD ruled out. Diabetes mellitus absent.",
        "expected": set(), "flag": False,
    },
    {
        "id": 19, "note": "[NEG] Negative for stroke, unlikely cancer",
        "text": "Negative for ischemic stroke. Metastatic cancer unlikely.",
        "expected": set(), "flag": False,
    },
    # ── SECTION filter: family/past history only — should NOT flag ────────────
    {
        "id": 20, "note": "[SEC] Family history only — DM + HTN",
        "text": "Family History: Diabetes mellitus, hypertension in mother.",
        "expected": set(), "flag": False,
    },
    {
        "id": 21, "note": "[SEC] PMH only — ESRD in father",
        "text": "Past Medical History: ESRD in father. No current renal disease.",
        "expected": set(), "flag": False,
    },
    # ── MIXED: negated + active in same note ─────────────────────────────────
    {
        "id": 22, "note": "[MIX] Negated CHF but active HTN + ESRD",
        "text": "No heart failure. Patient has hypertension and ESRD.",
        "expected": {"HYPERTENSION", "KIDNEY_DISEASE"}, "flag": True,
    },
    {
        "id": 23, "note": "[MIX] Family HTN but current CKD stage 4",
        "text": "Family History: Hypertension. Assessment: CKD stage 4, eGFR 28.",
        "expected": {"KIDNEY_DISEASE", "KIDNEY_FUNCTION"}, "flag": True,
    },
    # ── STAGE & eGFR inference ────────────────────────────────────────────────
    {
        "id": 24, "note": "[STAGE] CKD stage 3b explicit",
        "text": "Patient has CKD stage 3b. No dialysis at this time.",
        "expected": {"KIDNEY_DISEASE"}, "flag": True,
    },
    {
        "id": 25, "note": "[eGFR] eGFR=18 -> Stage 4 CKD inferred",
        "text": "eGFR = 18 mL/min. Referral for dialysis planning.",
        "expected": {"KIDNEY_FUNCTION", "DIALYSIS"}, "flag": True,
    },
    # ── MULTI-CONDITION complex notes ─────────────────────────────────────────
    {
        "id": 26, "note": "[MULTI] 5 conditions in one note",
        "text": (
            "Patient with ESRD on hemodialysis, type 2 diabetes mellitus, "
            "hypertension, and COPD. Palliative intent discussed."
        ),
        "expected": {
            "KIDNEY_DISEASE", "DIALYSIS", "DIABETES",
            "HYPERTENSION", "LUNG_DISEASE", "PALLIATIVE"
        },
        "flag": True,
    },
    {
        "id": 27, "note": "[MULTI] Hospice + comfort care + DNR + CKD",
        "text": (
            "Patient enrolled in hospice. Comfort care measures only. "
            "DNR documented. CKD stage 5."
        ),
        "expected": {"HOSPICE", "COMFORT_CARE", "END_OF_LIFE", "KIDNEY_DISEASE"},
        "flag": True,
    },
    # ── EDGE CASES ────────────────────────────────────────────────────────────
    {
        "id": 28, "note": "[EDGE] Temporal: history-of MI but current HTN",
        "text": "History of myocardial infarction. Currently has hypertension.",
        "expected": {"HEART_DISEASE", "HYPERTENSION"}, "flag": True,
    },
    {
        "id": 29, "note": "[EDGE] Completely healthy note",
        "text": "Annual wellness visit. Vaccines updated. No complaints.",
        "expected": set(), "flag": False,
    },
    {
        "id": 30, "note": "[EDGE] Renal transplant + eGFR normal",
        "text": "Renal transplant status. eGFR 68 post-transplant, stable.",
        "expected": {"TRANSPLANT", "KIDNEY_FUNCTION"}, "flag": True,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
config = PipelineConfig(
    exclude_negated=True,
    exclude_sections=["FAMILY_HISTORY", "PAST_HISTORY", "SOCIAL_HISTORY"],
    batch_log_every=9999,
)
pipeline = EnhancedTextMiningPipeline(config)

records = [
    {"DOCUMENT_ID": tc["id"], "CONTACT_LINE": 1,
     "CONTACT_DATE": "2026-01-01", "RESULT_TEXT": tc["text"]}
    for tc in TEST_CASES
]
results = pipeline.run(records, log_every=9999)
result_map = {r["document_id"]: r for r in results}

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 82)
print("  DETECTION ACCURACY EVALUATION — 30 GROUND-TRUTH RECORDS")
print("=" * 82)
print(f"  {'ID':<4} {'RESULT':<22} {'NOTE'}")
print(f"  {'-'*4} {'-'*22} {'-'*50}")

TP = FP = FN = TN = 0
label_tp = label_fn = label_fp = 0
detail_rows = []

for tc in TEST_CASES:
    r = result_map.get(tc["id"])
    was_flagged = r is not None
    detected_labels = {e["label"] for e in r["entities"]} if r else set()
    # Only count medical labels for comparison
    MEDICAL = {"KIDNEY_DISEASE","DIALYSIS","TRANSPLANT","KIDNEY_FUNCTION",
               "HOSPICE","PALLIATIVE","COMFORT_CARE","END_OF_LIFE",
               "HEART_DISEASE","LUNG_DISEASE","DIABETES","CANCER",
               "STROKE","HYPERTENSION","ICD_CODE"}
    detected_medical = detected_labels & MEDICAL
    expected = tc["expected"]

    # Record-level
    if tc["flag"] and was_flagged:
        TP += 1; status = "TP  CORRECT   "
    elif tc["flag"] and not was_flagged:
        FN += 1; status = "FN  MISSED    "
    elif not tc["flag"] and was_flagged:
        FP += 1; status = "FP  FALSE FLAG"
    else:
        TN += 1; status = "TN  CORRECT   "

    # Label-level
    found   = expected & detected_medical
    missed  = expected - detected_medical
    extra   = detected_medical - expected  # extra labels found (not wrong, just bonus)
    label_tp += len(found)
    label_fn += len(missed)

    icon = "OK" if status.startswith(("TP", "TN")) else "!!"
    print(f"  [{icon}] {tc['id']:<3} {status}  {tc['note']}")
    if missed:
        print(f"           !! Missed labels: {sorted(missed)}")
    if status.startswith("FP"):
        print(f"           !! Wrongly detected: {sorted(detected_medical)}")
    if status.startswith("FN") and detected_medical:
        print(f"           Detected anyway: {sorted(detected_medical)}")

    detail_rows.append({
        "id": tc["id"], "status": status, "expected": expected,
        "detected": detected_medical, "missed": missed, "extra": extra,
    })

total = len(TEST_CASES)
should_flag_total = sum(1 for tc in TEST_CASES if tc["flag"])
should_not_total  = total - should_flag_total

precision  = TP / (TP + FP) if (TP + FP) else 1.0
recall     = TP / (TP + FN) if (TP + FN) else 1.0
f1         = 2*precision*recall / (precision+recall) if (precision+recall) else 0
specificity = TN / (TN + FP) if (TN + FP) else 1.0
accuracy   = (TP + TN) / total

label_recall    = label_tp / (label_tp + label_fn) if (label_tp + label_fn) else 1.0
total_expected_labels = sum(len(tc["expected"]) for tc in TEST_CASES)

print()
print("=" * 82)
print("  RECORD-LEVEL RESULTS")
print(f"  Total records     : {total}  ({should_flag_total} positive / {should_not_total} negative)")
print(f"  True Positives    : {TP}  (flagged correctly)")
print(f"  True Negatives    : {TN}  (not flagged correctly)")
print(f"  False Positives   : {FP}  (flagged when should not be)")
print(f"  False Negatives   : {FN}  (missed when should be flagged)")
print()
print(f"  Precision         : {precision:.1%}  (of flagged records, how many were right)")
print(f"  Recall            : {recall:.1%}  (of positive records, how many were caught)")
print(f"  F1 Score          : {f1:.1%}  (harmonic mean)")
print(f"  Specificity       : {specificity:.1%}  (of negative records, how many left clean)")
print(f"  Overall Accuracy  : {accuracy:.1%}")
print()
print("  LABEL-LEVEL RESULTS")
print(f"  Total expected labels : {total_expected_labels}")
print(f"  Labels detected (TP)  : {label_tp}")
print(f"  Labels missed   (FN)  : {label_fn}")
print(f"  Label Recall          : {label_recall:.1%}  (fraction of conditions actually found)")
print("=" * 82)

if FN > 0:
    print()
    print("  MISSED CASES (investigate):")
    for r in detail_rows:
        if r["status"].startswith("FN"):
            print(f"  Doc {r['id']}: expected={sorted(r['expected'])} detected={sorted(r['detected'])}")
if FP > 0:
    print()
    print("  FALSE POSITIVE CASES:")
    for r in detail_rows:
        if r["status"].startswith("FP"):
            print(f"  Doc {r['id']}: wrongly detected={sorted(r['detected'])}")
if label_fn > 0:
    print()
    print("  MISSED LABELS:")
    for r in detail_rows:
        if r["missed"]:
            print(f"  Doc {r['id']}: missed={sorted(r['missed'])} (extra={sorted(r['extra'])})")
print("=" * 82)
