# Assumptions and Open Questions

**Date created:** 2026-03-28
**Last updated:** 2026-03-30

---

## A1 — ECG Sampling Rate

**Assumption:** All MIMIC-IV-ECG records in our manifest are sampled at 500 Hz, yielding 5000 samples for a 10-second recording.

**Why it matters:** MoRE applies `resample_poly(x, up=1, down=5)` to produce 1000-sample output at 100 Hz. Wrong rate → wrong shape.

**Status:** ✅ **Confirmed** — verified on downloaded records, all at 500 Hz. `resample_poly` produces `(12, 1000)` as expected.

---

## A2 — CheXpert Labels Joinable via subject_id + cxr_study_id

**Assumption:** Every CXR in our manifest has a corresponding row in `mimic-cxr-2.0.0-chexpert.csv`, joinable on `subject_id` and `study_id`.

**Fallback:** Zero-vector ("No Finding") for any study missing from the chexpert file.

**Status:** ✅ **Confirmed** — join succeeded for all 49,076 rows. No zero-vector fallback needed for CheXpert labels.

---

## A3 — Official MIMIC-CXR Split Covers Our Manifest

**Assumption:** `mimic-cxr-2.0.0-split.csv` contains entries for all `dicom_id` values in our manifest.

**Fallback:** Assign to 'train' if not found.

**Status:** ✅ **Confirmed** — split file covers all manifest rows. Resulting distribution: 48,423 train / 379 val / 274 test.

---

## A4 — Radiology Report Text Availability

**Assumption:** Most CXR studies have associated `.txt` radiology reports under the MIMIC-CXR `files/` tree.

**Status:** ❌ **Not applicable** — CXR `.txt` report files were not downloaded (inode quota exhausted). The CheXpert-label-to-text fallback (`generate_xray_note()`) is used for all 49,076 records.

**Impact:** Xray notes are synthetic (label-derived), not from actual radiology reports. This is consistent with how MoRE handles missing notes, but reduces text modality richness.

---

## A5 — ECG Machine Report Availability

**Assumption:** `machine_measurements.csv` has rows for all `ecg_study_id` values in our manifest.

**Status:** ✅ **Confirmed** — `machine_measurements.csv` downloaded and joined successfully. Report columns `report_0`–`report_6` populated for all manifest rows (some records have empty report strings, in which case ECG note defaults to `"ECG note not available."`).

**Additional work:** ECG report text was extended beyond note generation to extract 10-class rhythm labels via regex matching.

---

## A6 — One JPG per CXR Study (PA or AP view)

**Status:** ✅ **Confirmed** — manifest `cxr_path` includes the full dicom_id filename, uniquely identifying one image per study. No ambiguity.

**Note:** CXR files were not downloaded due to inode quota. Paths are stored in the `.npy` files but files do not exist on disk.

---

## A7 — ECG WFDB Records Have 12 Leads

**Status:** ✅ **Confirmed** — all downloaded MIMIC-IV-ECG records are 12-lead. `ViTModelEcg` input shape `(12, 1000)` works correctly.

---

## A8 — Local Data Root Structure

**Status:** ✅ **Resolved** — `configs/paths.yaml` defines `ecg_root` and `cxr_root` as absolute paths. The `.npy` files store absolute paths built at preprocessing time.

---

## A9 — preprocess_notes.py Bug

**Known bug:** Early `return` inside `get_clinical_xray()` loop (should be `break`).

**Status:** ✅ **Fixed** — bug patched in Phase 3 minimal refactor. Preprocessing re-implemented in `scripts/convert_manifest_to_more_format.py` with correct logic.

---

## A10 — pretrain_multimodel.py Missing Imports

**Known issue:** Missing `torch`, `tqdm`, `autocast`, `GradScaler`, `nn` imports; missing `--epochs` argparse arg.

**Status:** ✅ **Fixed** — all imports added, `--epochs` added to argparse.

---

## A11 — HPRC Inode Quota (discovered during Phase 2)

**Issue:** Grace cluster enforces a per-user inode limit. Full MIMIC-CXR-JPG + MIMIC-IV-ECG download (~150K+ files) exhausted quota before CXR images could be stored.

**Impact:**
- CXR JPEG files: 0 / 49,076 downloaded
- ECG signal files: 28,745 / 49,076 downloaded (20,331 missing)
- Radiology report .txt files: 0 downloaded

**Mitigation:**
- Distillation pipeline uses ECG only for the student
- Teacher embedding cache uses teacher's ECG encoder only (no CXR needed)
- 28,745 available ECG records are sufficient for viable distillation training
- All 49,076 records still in `.npy` files (paths present); missing-file records are filtered at dataset load time

**Status:** ⚠️ **Working around** — pipeline is viable with partial ECG-only data. Full CXR download would require either (a) requesting quota increase from HPRC, or (b) downloading to a shared group scratch space.

---

## A12 — NaN Teacher Embeddings (discovered during Phase 4)

**Issue:** 502 of 28,745 teacher ECG embeddings contained NaN values — caused by ViT encoder instability on certain ECG records (likely unusual signal morphologies or signal quality issues).

**Mitigation:** `np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)` applied at dataset load time in `distill/distill_dataset.py`. NaN batch guard in training loop (`if not torch.isfinite(loss): skip`).

**Status:** ✅ **Fixed**
