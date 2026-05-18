# datasets/fusion_dataset.py

import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import Counter

from configs.config import *
from datasets.skeleton_parser import (
    parse_skeleton_file,
    get_label_from_filename
)


class FusionDataset(Dataset):
    def __init__(
        self,
        cropped_video_paths,
        skeleton_root,
        num_frames=24,
        transform=None,
        is_train=True
    ):
        self.num_frames = num_frames
        self.transform = transform
        self.is_train = is_train

        self.samples = []
        self.node_features_list = []

        for vp in cropped_video_paths:

            video_name = os.path.splitext(
                os.path.basename(vp)
            )[0]

            # =========================
            # 标签
            # =========================
            try:
                label = get_label_from_filename(video_name)

            except ValueError:
                print(f"跳过无法解析标签的视频: {vp}")
                continue

            # =========================
            # 视频读取
            # =========================
            cap = cv2.VideoCapture(vp)

            if not cap.isOpened():
                print(f"无法打开视频: {vp}")
                continue

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            cap.release()

            if total_frames == 0:
                continue

            # =========================
            # 骨架文件
            # =========================
            skeleton_path = os.path.join(
                skeleton_root,
                video_name + ".skeleton"
            )

            if not os.path.exists(skeleton_path):
                print(f"骨架不存在: {skeleton_path}")
                continue

            skeleton_frames = parse_skeleton_file(
                skeleton_path
            )

            T_skel = len(skeleton_frames)

            if T_skel == 0:
                continue

            T = min(total_frames, T_skel)

            # =========================
            # 全局归一化
            # =========================
            all_joints = []

            for frame_joints in skeleton_frames[:T]:

                for (x, y) in frame_joints:

                    if x > 0 and y > 0:
                        all_joints.append((x, y))

            if len(all_joints) == 0:
                print(f"无有效关节: {video_name}")
                continue

            xs = [p[0] for p in all_joints]
            ys = [p[1] for p in all_joints]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # =========================
            # 节点特征
            # =========================
            node_feats_seq = []

            for t in range(T):

                joints_raw = skeleton_frames[t]

                feats = []

                for (x, y) in joints_raw:

                    if x > 0 and y > 0:

                        nx = (x - x_min) / (
                            x_max - x_min + 1e-8
                        )

                        ny = (y - y_min) / (
                            y_max - y_min + 1e-8
                        )

                        nx = np.clip(nx, 0, 1)
                        ny = np.clip(ny, 0, 1)

                    else:
                        nx, ny = 0.0, 0.0

                    feats.append([nx, ny])

                node_feats_seq.append(
                    np.array(feats, dtype=np.float32)
                )

            node_feats_arr = np.stack(
                node_feats_seq,
                axis=0
            )

            self.node_features_list.append(
                node_feats_arr
            )

            self.samples.append(
                (vp, label, total_frames)
            )

        # =========================
        # 数据集统计
        # =========================
        labels = [s[1] for s in self.samples]

        if len(labels) > 0:

            frame_counts = [s[2] for s in self.samples]

            print(f"数据集样本数: {len(self.samples)}")
            print("标签分布:", Counter(labels))

            print(
                f"帧数: "
                f"min={min(frame_counts)}, "
                f"max={max(frame_counts)}, "
                f"mean={np.mean(frame_counts):.1f}"
            )

        else:
            print("数据集为空！")

    # ==================================================
    # 采样帧
    # ==================================================
    def _sample_frames(self, total_frames):

        if self.is_train:

            if total_frames < self.num_frames:

                indices = list(range(total_frames))

                pad = [total_frames - 1] * (
                    self.num_frames - total_frames
                )

                return indices + pad

            else:

                segment_size = (
                    total_frames / self.num_frames
                )

                indices = []

                for i in range(self.num_frames):

                    start = int(i * segment_size)

                    end = int(
                        (i + 1) * segment_size
                    )

                    if end > total_frames:
                        end = total_frames

                    if start >= end:
                        start = max(0, end - 1)

                    rand_idx = (
                        np.random.randint(start, end)
                        if start < end
                        else start
                    )

                    indices.append(rand_idx)

                return indices

        else:

            if total_frames < self.num_frames:

                indices = list(range(total_frames))

                pad = [total_frames - 1] * (
                    self.num_frames - total_frames
                )

                return indices + pad

            else:

                step = total_frames / self.num_frames

                return [
                    int(i * step)
                    for i in range(self.num_frames)
                ]

    # ==================================================
    # 长度
    # ==================================================
    def __len__(self):
        return len(self.samples)

    # ==================================================
    # 获取样本
    # ==================================================
    def __getitem__(self, idx):

        video_path, label, total_frames = (
            self.samples[idx]
        )

        frame_indices = self._sample_frames(
            total_frames
        )

        # =========================
        # RGB
        # =========================
        cap = cv2.VideoCapture(video_path)

        frames = []

        for fidx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                fidx
            )

            ret, frame = cap.read()

            if not ret:

                frame = np.zeros(
                    (img_size, img_size, 3),
                    dtype=np.uint8
                )

                pil_img = Image.fromarray(frame)

            else:

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                pil_img = Image.fromarray(
                    frame_rgb
                )

            if self.transform:

                img_tensor = self.transform(
                    pil_img
                )

            else:

                img_tensor = transforms.ToTensor()(
                    pil_img
                )

            frames.append(img_tensor)

        cap.release()

        frames_tensor = torch.stack(
            frames,
            dim=0
        )

        # =========================
        # Skeleton
        # =========================
        node_feats = self.node_features_list[idx][
            frame_indices
        ]

        node_feats_tensor = torch.from_numpy(
            node_feats
        ).float()

        return (
            frames_tensor,
            node_feats_tensor,
            torch.tensor(label, dtype=torch.long)
        )


# ======================================================
# collate
# ======================================================
def collate_fn_fusion(batch):

    frames = torch.stack(
        [b[0] for b in batch],
        dim=0
    )

    node_feats = torch.stack(
        [b[1] for b in batch],
        dim=0
    )

    labels = torch.stack(
        [b[2] for b in batch],
        dim=0
    )

    return frames, node_feats, labels