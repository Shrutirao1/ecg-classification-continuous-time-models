from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm.auto import tqdm


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_beats(
    model,
    loader,
    criterion,
    device: str,
    show_progress: bool = False,
    progress_desc: str = "Validation",
) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_preds: list[float] = []
    all_labels: list[float] = []

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=progress_desc, unit="batch", leave=True)

    with torch.no_grad():
        for inputs, labels, _ in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    probs_np = np.asarray(all_probs)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(labels_np, preds_np),
        "f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "roc_auc": _safe_roc_auc(labels_np, probs_np),
        "labels": labels_np,
        "preds": preds_np,
        "probs": probs_np,
        "confusion_matrix": cm,
    }


def evaluate_subjects(model, loader, criterion, device: str, cutoff: float = 0.5) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    subject_probs: dict[str, list[float]] = defaultdict(list)
    subject_labels: dict[str, float] = {}

    with torch.no_grad():
        for inputs, labels, subject_ids in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for idx, subject_id in enumerate(subject_ids):
                subject_probs[str(subject_id)].append(float(probs[idx]))
                subject_labels[str(subject_id)] = float(labels_np[idx])

    aggregated_scores: list[float] = []
    aggregated_preds: list[int] = []
    aggregated_labels: list[int] = []
    for subject_id, probs in subject_probs.items():
        beat_preds = (np.asarray(probs) > 0.5).astype(int)
        pos_ratio = float(beat_preds.mean())
        aggregated_scores.append(pos_ratio)
        aggregated_preds.append(int(pos_ratio > cutoff))
        aggregated_labels.append(int(subject_labels[subject_id]))

    labels_np = np.asarray(aggregated_labels)
    preds_np = np.asarray(aggregated_preds)
    scores_np = np.asarray(aggregated_scores)
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(labels_np, preds_np),
        "f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "roc_auc": _safe_roc_auc(labels_np, scores_np),
        "labels": labels_np,
        "scores": scores_np,
        "preds": preds_np,
        "confusion_matrix": cm,
        "cutoff": cutoff,
    }


def find_best_subject_cutoff(
    model,
    loader,
    criterion,
    device: str,
    cutoffs: np.ndarray | None = None,
) -> tuple[float, dict[str, float | np.ndarray]]:
    if cutoffs is None:
        cutoffs = np.arange(0.1, 0.95, 0.05)

    best_cutoff = 0.5
    best_metrics: dict[str, float | np.ndarray] | None = None

    for cutoff in cutoffs:
        metrics = evaluate_subjects(model, loader, criterion, device=device, cutoff=float(cutoff))
        if best_metrics is None or float(metrics["f1"]) > float(best_metrics["f1"]):
            best_cutoff = float(cutoff)
            best_metrics = metrics

    assert best_metrics is not None
    return best_cutoff, best_metrics


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: str,
    epochs: int,
    checkpoint_path: str | Path,
    scheduler=None,
    verbose: bool = True,
    show_epoch_progress: bool = True,
    show_batch_progress: bool = False,
) -> tuple[object, list[dict[str, float]]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history: list[dict[str, float]] = []
    epoch_iterator = range(epochs)
    if show_epoch_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="Training", unit="epoch")

    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0

        batch_iterator = train_loader
        if show_batch_progress:
            batch_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=True,
            )

        for batch_idx, (inputs, labels, _) in enumerate(batch_iterator, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if show_batch_progress:
                batch_iterator.set_postfix(
                    batch_loss=f"{loss.item():.4f}",
                    avg_loss=f"{(running_loss / batch_idx):.4f}",
                )

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate_beats(
            model,
            val_loader,
            criterion,
            device=device,
            show_progress=show_batch_progress,
            progress_desc=f"Validation {epoch + 1}/{epochs}",
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["f1"]),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
        }
        history.append(epoch_metrics)

        if show_epoch_progress:
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{epoch_metrics['val_loss']:.4f}",
                val_f1=f"{epoch_metrics['val_f1']:.4f}",
            )

        if verbose:
            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={epoch_metrics['train_loss']:.4f} | "
                f"val_loss={epoch_metrics['val_loss']:.4f} | "
                f"val_acc={epoch_metrics['val_accuracy']:.4f} | "
                f"val_f1={epoch_metrics['val_f1']:.4f} | "
                f"val_sens={epoch_metrics['val_sensitivity']:.4f} | "
                f"val_spec={epoch_metrics['val_specificity']:.4f} | "
                f"val_auc={epoch_metrics['val_roc_auc']:.4f}"
            )
            if show_epoch_progress or show_batch_progress:
                tqdm.write(message)
            else:
                print(message, flush=True)

        if scheduler is not None:
            scheduler.step(float(val_metrics["loss"]))

        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history
