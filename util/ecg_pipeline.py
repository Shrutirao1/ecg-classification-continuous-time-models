from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import biosppy.signals.ecg as biosppy_ecg
import numpy as np
import pandas as pd
import pywt
import torch
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


TARGET_FS = 125
FIXED_BEAT_LENGTH = 150
PTBXL_DATASET_NAME = "ptbxl"
PTBDB_DATASET_NAME = "ptbdb"


def resample_signal(signal: np.ndarray, original_fs: int | float, target_fs: int = TARGET_FS) -> np.ndarray:
    if int(original_fs) == int(target_fs):
        return signal
    original_fs = int(original_fs)
    target_fs = int(target_fs)
    gcd = math.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd
    return resample_poly(signal, up=up, down=down, axis=0)


def normalize_and_fix_length(signal: np.ndarray, fixed_length: int = FIXED_BEAT_LENGTH) -> np.ndarray:
    normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
    if len(normalized) < fixed_length:
        normalized = np.concatenate((normalized, np.zeros(fixed_length - len(normalized), dtype=normalized.dtype)))
    elif len(normalized) > fixed_length:
        normalized = normalized[:fixed_length]
    return normalized.astype(np.float32)


def extract_beat_from_segment(ecg_segment: np.ndarray, fs: int, fixed_length: int = FIXED_BEAT_LENGTH) -> np.ndarray | None:
    try:
        out = biosppy_ecg.ecg(signal=ecg_segment, sampling_rate=fs, show=False)
        filtered_signal = out["filtered"]
        rpeaks = out["rpeaks"]
    except Exception:
        return None

    if len(rpeaks) < 2:
        return None

    rr_intervals = rpeaks[1:] - rpeaks[:-1]
    rr_median = np.median(rr_intervals)
    median_idx = int(np.argsort(np.abs(rr_intervals - rr_median))[0])
    median_signal_idx = int(rpeaks[median_idx + 1])

    half_window = int(rr_median * 0.6)
    start = max(0, median_signal_idx - half_window)
    end = min(len(filtered_signal), median_signal_idx + half_window)
    template_signal = filtered_signal[start:end]

    if median_signal_idx < len(filtered_signal) and filtered_signal[median_signal_idx] < 0:
        template_signal = -template_signal

    if len(template_signal) == 0:
        return None

    return normalize_and_fix_length(template_signal, fixed_length=fixed_length)


def extract_all_beats_from_signal(signal: np.ndarray, fs: int, fixed_length: int = FIXED_BEAT_LENGTH) -> list[np.ndarray]:
    beats: list[np.ndarray] = []
    segment_length = 10 * fs
    total_segments = len(signal) // segment_length

    for idx in range(total_segments):
        segment = signal[idx * segment_length : (idx + 1) * segment_length]
        beat = extract_beat_from_segment(segment, fs=fs, fixed_length=fixed_length)
        if beat is not None:
            beats.append(beat)

    return beats


def compute_dwt(signal: np.ndarray) -> np.ndarray:
    coeff_a, coeff_d = pywt.dwt(signal, "db4")
    return np.concatenate([coeff_a, coeff_d]).astype(np.float32)


def _label_name_to_int(label_name: str) -> int:
    return 1 if label_name == "patient" else 0


def _parse_ptbdb_reason_for_admission(header_path: Path) -> str | None:
    text = header_path.read_text(errors="ignore")
    match = re.search(r"reason for admission:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower()


def _collect_ptbdb_subject_labels(root_dir: Path) -> dict[str, str]:
    controls_path = root_dir / "CONTROLS"
    controls = {
        line.strip().split("/")[0]
        for line in controls_path.read_text().splitlines()
        if line.strip()
    }

    subject_labels: dict[str, str] = {}
    for patient_dir in sorted(root_dir.glob("patient*")):
        subject_id = patient_dir.name
        if subject_id in controls:
            subject_labels[subject_id] = "control"
            continue

        keep_subject = True
        found_mi = False
        for header_path in sorted(patient_dir.glob("*.hea")):
            diagnosis = _parse_ptbdb_reason_for_admission(header_path)
            if diagnosis is None:
                continue
            if "myocardial infarction" not in diagnosis:
                keep_subject = False
                break
            found_mi = True

        if keep_subject and found_mi:
            subject_labels[subject_id] = "patient"

    return subject_labels


def _save_beats(
    beats: Iterable[np.ndarray],
    output_dir: Path,
    dataset_name: str,
    subject_id: str,
    record_id: str,
    label_name: str,
    metadata_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    next_index = len(metadata_rows)
    for beat_idx, beat in enumerate(beats):
        file_name = f"{dataset_name}_{next_index:08d}.npy"
        file_path = output_dir / file_name
        np.save(file_path, beat)
        metadata_rows.append(
            {
                "dataset": dataset_name,
                "subject_id": subject_id,
                "record_id": record_id,
                "beat_idx": beat_idx,
                "label_name": label_name,
                "label": _label_name_to_int(label_name),
                "file_path": str(file_path.resolve()),
            }
        )
        next_index += 1


def preprocess_ptbdb_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    target_fs: int = TARGET_FS,
) -> pd.DataFrame:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    subject_labels = _collect_ptbdb_subject_labels(root_dir)
    metadata_rows: list[dict[str, object]] = []

    for patient_dir in sorted(root_dir.glob("patient*")):
        subject_key = patient_dir.name
        if subject_key not in subject_labels:
            continue

        label_name = subject_labels[subject_key]
        subject_id = f"{PTBDB_DATASET_NAME}_{subject_key}"

        for header_path in sorted(patient_dir.glob("*.hea")):
            record_base = header_path.with_suffix("")
            record = wfdb.rdrecord(str(record_base))
            signal = resample_signal(record.p_signal, record.fs, target_fs)
            lead_ii = signal[:, 1]
            beats = extract_all_beats_from_signal(lead_ii, fs=target_fs)
            record_id = record_base.name
            _save_beats(
                beats=beats,
                output_dir=output_dir,
                dataset_name=PTBDB_DATASET_NAME,
                subject_id=subject_id,
                record_id=record_id,
                label_name=label_name,
                metadata_rows=metadata_rows,
            )

    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata = metadata.sort_values(["subject_id", "record_id", "beat_idx"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def _build_ptbxl_record_table(root_dir: Path) -> pd.DataFrame:
    database = pd.read_csv(root_dir / "ptbxl_database.csv")
    scp_statements = pd.read_csv(root_dir / "scp_statements.csv", index_col=0)
    mi_codes = set(scp_statements.index[scp_statements["diagnostic_class"] == "MI"])

    def record_label(scp_code_text: str) -> str | None:
        codes = ast.literal_eval(scp_code_text)
        code_keys = set(codes.keys())
        if code_keys & mi_codes:
            return "patient"

        diagnostic_classes = {
            str(scp_statements.loc[code, "diagnostic_class"])
            for code in code_keys
            if code in scp_statements.index and pd.notna(scp_statements.loc[code, "diagnostic_class"])
        }
        if "NORM" in code_keys and diagnostic_classes <= {"NORM"}:
            return "control"
        return None

    database["record_label"] = database["scp_codes"].apply(record_label)
    database = database.dropna(subset=["record_label"]).copy()

    keep_rows: list[pd.DataFrame] = []
    for patient_id, group in database.groupby("patient_id"):
        labels = set(group["record_label"])
        if "patient" in labels:
            keep_rows.append(group[group["record_label"] == "patient"])
        elif labels == {"control"}:
            keep_rows.append(group)

    if not keep_rows:
        return pd.DataFrame(columns=database.columns)

    filtered = pd.concat(keep_rows, ignore_index=True)
    filtered["subject_id"] = filtered["patient_id"].apply(lambda value: f"{PTBXL_DATASET_NAME}_{int(value)}")
    return filtered


def preprocess_ptbxl_dataset(
    root_dir: str | Path,
    output_dir: str | Path,
    target_fs: int = TARGET_FS,
    use_high_resolution: bool = True,
) -> pd.DataFrame:
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    record_table = _build_ptbxl_record_table(root_dir)
    metadata_rows: list[dict[str, object]] = []

    filename_column = "filename_hr" if use_high_resolution else "filename_lr"
    expected_fs = 500 if use_high_resolution else 100

    for row in record_table.itertuples(index=False):
        record_base = root_dir / getattr(row, filename_column)
        record = wfdb.rdrecord(str(record_base))
        signal = resample_signal(record.p_signal, record.fs, target_fs)
        lead_ii = signal[:, 1]
        beats = extract_all_beats_from_signal(lead_ii, fs=target_fs)
        _save_beats(
            beats=beats,
            output_dir=output_dir,
            dataset_name=PTBXL_DATASET_NAME,
            subject_id=row.subject_id,
            record_id=record_base.name,
            label_name=row.record_label,
            metadata_rows=metadata_rows,
        )

    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata["source_fs"] = expected_fs
        metadata["target_fs"] = target_fs
        metadata = metadata.sort_values(["subject_id", "record_id", "beat_idx"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def summarize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(
            [
                {
                    "beats": 0,
                    "subjects": 0,
                    "records": 0,
                }
            ]
        )

    return (
        metadata.groupby("label_name")
        .agg(beats=("file_path", "count"), subjects=("subject_id", "nunique"), records=("record_id", "nunique"))
        .reset_index()
    )


class BeatDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, use_dwt: bool = False):
        self.metadata = metadata.reset_index(drop=True)
        self.use_dwt = use_dwt

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.metadata.iloc[idx]
        data = np.load(row["file_path"]).astype(np.float32)
        if self.use_dwt:
            data = compute_dwt(data)
        tensor = torch.from_numpy(data).float().unsqueeze(0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        subject_id = str(row["subject_id"])
        return tensor, label, subject_id


@dataclass
class DataSplit:
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame


def split_subjects_for_validation(
    metadata: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    subject_frame = metadata.groupby("subject_id")["label"].first().reset_index()
    train_subjects, val_subjects = train_test_split(
        subject_frame["subject_id"],
        test_size=val_size,
        stratify=subject_frame["label"],
        random_state=random_state,
    )

    train_metadata = metadata[metadata["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_metadata = metadata[metadata["subject_id"].isin(val_subjects)].reset_index(drop=True)
    return DataSplit(train_metadata=train_metadata, val_metadata=val_metadata)


def build_dataloaders(
    train_metadata_path: str | Path,
    batch_size: int = 64,
    val_size: float = 0.2,
    random_state: int = 42,
    use_dwt: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    train_metadata = pd.read_csv(train_metadata_path)
    split = split_subjects_for_validation(
        metadata=train_metadata,
        val_size=val_size,
        random_state=random_state,
    )

    train_dataset = BeatDataset(split.train_metadata, use_dwt=use_dwt)
    val_dataset = BeatDataset(split.val_metadata, use_dwt=use_dwt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, split.train_metadata, split.val_metadata


def build_test_loader(
    metadata_path: str | Path,
    batch_size: int = 64,
    use_dwt: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    dataset = BeatDataset(metadata, use_dwt=use_dwt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, metadata

