# trainers/trainer.py

import torch
import torch.nn as nn

from tqdm import tqdm
from collections import Counter

# =========================================================
# Config
# =========================================================
from configs.config import *

# =========================================================
# Model
# =========================================================
from models.fusion_model import FusionModel

# =========================================================
# Metrics
# =========================================================
from utils.metrics import calculate_metrics


# =========================================================
# Train One Fold
# =========================================================
def train_fold(
    train_loader,
    val_loader,
    fold_idx
):

    # =====================================================
    # Model
    # =====================================================
    model = FusionModel(
        num_classes=num_classes
    ).to(device)

    # =====================================================
    # Compute Class Weights
    # =====================================================
    train_labels = []

    for sample in train_loader.dataset.samples:

        train_labels.append(sample[1])

    class_counts = Counter(train_labels)

    class_weights = torch.tensor(
        [
            1.0 / (class_counts[0] + 1e-8),
            1.0 / (class_counts[1] + 1e-8)
        ],
        dtype=torch.float
    ).to(device)

    class_weights = (
        class_weights
        / class_weights.sum()
        * 2
    )

    print(f"\nClass Weights: {class_weights}")

    # =====================================================
    # Optimizer
    # =====================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # =====================================================
    # Scheduler
    # =====================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # =====================================================
    # Loss
    # =====================================================
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # =====================================================
    # AMP
    # =====================================================
    if use_amp:

        scaler = torch.amp.GradScaler('cuda')

    else:

        scaler = None

    # =====================================================
    # Early Stop
    # =====================================================
    best_acc = 0.0

    best_state_dict = None

    patience = 8

    no_improve = 0

    # =====================================================
    # Epoch Loop
    # =====================================================
    for epoch in range(epochs):

        # =================================================
        # Train
        # =================================================
        model.train()

        total_loss = 0

        correct = 0

        total = 0

        optimizer.zero_grad()

        train_bar = tqdm(
            train_loader,
            desc=f"Fold {fold_idx+1} Epoch {epoch+1}"
        )

        for i, (frames, node_feats, labels) in enumerate(train_bar):

            frames = frames.to(device)

            node_feats = node_feats.to(device)

            labels = labels.to(device)

            # =============================================
            # Forward
            # =============================================
            if use_amp:

                with torch.amp.autocast('cuda'):

                    outputs = model(
                        frames,
                        node_feats
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )

                    loss = (
                        loss
                        / accumulation_steps
                    )

                scaler.scale(loss).backward()

            else:

                outputs = model(
                    frames,
                    node_feats
                )

                loss = criterion(
                    outputs,
                    labels
                )

                loss = (
                    loss
                    / accumulation_steps
                )

                loss.backward()

            # =============================================
            # Gradient Clip
            # =============================================
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # =============================================
            # Update
            # =============================================
            if (i + 1) % accumulation_steps == 0:

                if use_amp:

                    scaler.step(optimizer)

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad()

            # =============================================
            # Statistics
            # =============================================
            total_loss += (
                loss.item()
                * accumulation_steps
            )

            preds = outputs.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            train_bar.set_postfix({

                "loss":
                    f"{total_loss/(i+1):.4f}",

                "acc":
                    f"{correct/total:.4f}"
            })

        train_acc = correct / total

        print(
            f"\nFold {fold_idx+1} "
            f"| Epoch {epoch+1} "
            f"| Train Loss: {total_loss/len(train_loader):.4f} "
            f"| Train Acc: {train_acc:.4f}"
        )

        # =================================================
        # Validation
        # =================================================
        model.eval()

        val_loss = 0

        val_preds = []

        val_labels = []

        val_probs = []

        with torch.no_grad():

            for frames, node_feats, labels in val_loader:

                frames = frames.to(device)

                node_feats = node_feats.to(device)

                labels = labels.to(device)

                outputs = model(
                    frames,
                    node_feats
                )

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)

                val_preds.extend(
                    preds.cpu().numpy()
                )

                val_labels.extend(
                    labels.cpu().numpy()
                )

                val_probs.extend(
                    probs[:, 1].cpu().numpy()
                )

        # =================================================
        # Metrics
        # =================================================
        acc, prec, rec, f1, auc = calculate_metrics(

            val_labels,

            val_preds,

            val_probs
        )

        print(
            f"Fold {fold_idx+1} "
            f"| Epoch {epoch+1} "
            f"| Val Loss: {val_loss/len(val_loader):.4f} "
            f"| Acc: {acc:.4f} "
            f"| Prec: {prec:.4f} "
            f"| Rec: {rec:.4f} "
            f"| F1: {f1:.4f} "
            f"| AUC: {auc:.4f}"
        )

        # =================================================
        # Save Best
        # =================================================
        if acc > best_acc:

            best_acc = acc

            best_state_dict = model.state_dict().copy()

            no_improve = 0

        else:

            no_improve += 1

            if no_improve >= patience:

                print(
                    f"\nFold {fold_idx+1} "
                    f"Early stopping at epoch {epoch+1}"
                )

                break

        scheduler.step()

    # =====================================================
    # Load Best
    # =====================================================
    model.load_state_dict(best_state_dict)

    # =====================================================
    # Final Eval
    # =====================================================
    model.eval()

    val_preds = []

    val_labels = []

    val_probs = []

    with torch.no_grad():

        for frames, node_feats, labels in val_loader:

            frames = frames.to(device)

            node_feats = node_feats.to(device)

            labels = labels.to(device)

            outputs = model(
                frames,
                node_feats
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            val_preds.extend(
                outputs.argmax(dim=1).cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_probs.extend(
                probs[:, 1].cpu().numpy()
            )

    # =====================================================
    # Final Metrics
    # =====================================================
    final_acc, final_prec, final_rec, final_f1, final_auc = calculate_metrics(

        val_labels,

        val_preds,

        val_probs
    )

    # =====================================================
    # Save Model
    # =====================================================
    save_path = (
        f"{output_model_path}"
        f"_fold{fold_idx+1}.pth"
    )

    torch.save(
        best_state_dict,
        save_path
    )

    print("\n=================================================")

    print(
        f"Fold {fold_idx+1} Finished "
        f"| Best Acc: {best_acc:.4f}"
    )

    print(
        f"Final Metrics "
        f"| Acc={final_acc:.4f} "
        f"| Prec={final_prec:.4f} "
        f"| Rec={final_rec:.4f} "
        f"| F1={final_f1:.4f} "
        f"| AUC={final_auc:.4f}"
    )

    print(
        f"Checkpoint Saved: {save_path}"
    )

    print("=================================================\n")

    return (
        final_acc,
        final_prec,
        final_rec,
        final_f1,
        final_auc
    )