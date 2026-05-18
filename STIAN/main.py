# main.py

import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import KFold

from configs.config import *

from datasets.fusion_dataset import (
    FusionDataset,
    collate_fn_fusion
)

from trainers.trainer import train_fold

from utils.seed import set_seed


# ==========================================================
# Image Transform
# ==========================================================
train_transform = transforms.Compose([

    transforms.Resize(
        (img_size, img_size)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (img_size, img_size)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# Main
# ==========================================================
def main():

    # ======================================================
    # Seed
    # ======================================================
    set_seed(random_state)

    # ======================================================
    # Get video paths
    # ======================================================
    video_paths = [

        os.path.join(
            cropped_video_folder,
            f
        )

        for f in os.listdir(cropped_video_folder)

        if f.endswith(".mp4")
    ]

    print(f"找到 {len(video_paths)} 个视频")

    # ======================================================
    # KFold
    # ======================================================
    all_indices = np.arange(len(video_paths))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_metrics = []

    # ======================================================
    # Cross Validation
    # ======================================================
    for fold_idx, (train_idx, val_idx) in enumerate(

        kf.split(all_indices)

    ):

        print(
            f"\n========== Fold {fold_idx + 1} =========="
        )

        # ==================================================
        # Train / Val paths
        # ==================================================
        train_paths = [

            video_paths[i]

            for i in train_idx
        ]

        val_paths = [

            video_paths[i]

            for i in val_idx
        ]

        # ==================================================
        # Dataset
        # ==================================================
        train_dataset = FusionDataset(

            cropped_video_paths=train_paths,

            skeleton_root=skeleton_folder,

            num_frames=num_frames,

            transform=train_transform,

            is_train=True
        )

        val_dataset = FusionDataset(

            cropped_video_paths=val_paths,

            skeleton_root=skeleton_folder,

            num_frames=num_frames,

            transform=val_transform,

            is_train=False
        )

        # ==================================================
        # 防止空数据集
        # ==================================================
        if len(train_dataset) == 0:

            print("训练集为空，跳过该 Fold")

            continue

        if len(val_dataset) == 0:

            print("验证集为空，跳过该 Fold")

            continue

        # ==================================================
        # DataLoader
        # ==================================================
        train_loader = DataLoader(

            train_dataset,

            batch_size=batch_size,

            shuffle=True,

            collate_fn=collate_fn_fusion,

            num_workers=2,

            pin_memory=True
        )

        val_loader = DataLoader(

            val_dataset,

            batch_size=batch_size,

            shuffle=False,

            collate_fn=collate_fn_fusion,

            num_workers=2,

            pin_memory=True
        )

        # ==================================================
        # Train
        # ==================================================
        metrics = train_fold(

            train_loader,

            val_loader,

            fold_idx
        )

        fold_metrics.append(metrics)

    # ======================================================
    # Final Result
    # ======================================================
    if len(fold_metrics) == 0:

        print("没有成功训练的 Fold")

        return

    print("\n========== 五折交叉验证结果 ==========")

    avg_metrics = np.mean(
        fold_metrics,
        axis=0
    )

    std_metrics = np.std(
        fold_metrics,
        axis=0
    )

    print(
        f"平均 Acc: "
        f"{avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}"
    )

    print(
        f"平均 Prec: "
        f"{avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}"
    )

    print(
        f"平均 Rec: "
        f"{avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}"
    )

    print(
        f"平均 F1: "
        f"{avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}"
    )

    print(
        f"平均 AUC: "
        f"{avg_metrics[4]:.4f} ± {std_metrics[4]:.4f}"
    )


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":

    main()