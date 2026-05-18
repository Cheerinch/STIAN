# models/fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from configs.config import *

from models.gcn_encoder import GCNEncoder
from models.biased_transformer import (
    BiasedTransformerEncoderLayer
)

from utils.graph import (
    edge_index,
    part_indices_list
)


class FusionModel(nn.Module):

    def __init__(
        self,
        num_classes=2,
        rgb_dim=512,
        skel_dim=32,
        hidden_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # ======================================================
        # RGB Backbone
        # ======================================================
        self.rgb_backbone = timm.create_model(
            'resnet18',
            pretrained=True
        )

        self.rgb_backbone.reset_classifier(0)

        self.rgb_dim = rgb_dim

        # ======================================================
        # GCN Encoder
        # ======================================================
        self.gcn = GCNEncoder(
            in_dim=2,
            hidden_dim=64,
            out_dim=skel_dim
        )

        self.register_buffer(
            'edge_index',
            edge_index
        )

        # ======================================================
        # Part Indices
        # ======================================================
        self.part_indices = part_indices_list

        self.num_parts = len(self.part_indices)

        # ======================================================
        # Fusion Layer
        # ======================================================
        self.fc_fuse = nn.Linear(
            rgb_dim + skel_dim,
            hidden_dim
        )

        # ======================================================
        # Position Encoding
        # ======================================================
        self.pos_encoder = nn.Parameter(
            torch.randn(
                1,
                num_frames,
                hidden_dim
            ) * 0.02
        )

        # ======================================================
        # Transformer Layers
        # ======================================================
        self.transformer_layers = nn.ModuleList([

            BiasedTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ======================================================
        # Classifier
        # ======================================================
        self.classifier = nn.Sequential(

            nn.Dropout(0.6),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x, node_feats):

        B, T, C, H, W = x.shape

        if T < 2:
            raise ValueError(
                f"帧数必须 >=2，当前 T={T}"
            )

        # ======================================================
        # RGB Feature
        # ======================================================
        rgb_in = x.view(
            B * T,
            C,
            H,
            W
        )

        rgb_feat = self.rgb_backbone.forward_features(
            rgb_in
        )

        rgb_feat = rgb_feat.mean(dim=[2, 3])

        rgb_feat = rgb_feat.view(
            B,
            T,
            -1
        )

        # ======================================================
        # GCN Feature
        # ======================================================
        node_out_all = []

        for t in range(T):

            batch_node = []

            for b in range(B):

                x_node = node_feats[b, t]

                node_out = self.gcn(
                    x_node,
                    self.edge_index
                )

                batch_node.append(node_out)

            node_out_all.append(
                torch.stack(batch_node, dim=0)
            )

        node_out_all = torch.stack(
            node_out_all,
            dim=1
        )

        # shape:
        # (B, T, 17, skel_dim)

        # ======================================================
        # Part Features
        # ======================================================
        part_feats = []

        for indices in self.part_indices:

            indices = indices.to(
                node_out_all.device
            )

            part_nodes = node_out_all[
                :,
                :,
                indices,
                :
            ]

            # (B, T, L, skel_dim)

            part_avg = part_nodes.mean(dim=2)

            # (B, T, skel_dim)

            part_feats.append(part_avg)

        part_feats = torch.stack(
            part_feats,
            dim=2
        )

        # shape:
        # (B, T, num_parts, skel_dim)

        # ======================================================
        # Temporal Inconsistency
        # ======================================================
        part_probs = F.softmax(
            part_feats,
            dim=-1
        )

        kl_diff = torch.zeros(
            B,
            T - 1,
            self.num_parts,
            device=part_feats.device
        )

        for t in range(T - 1):

            p = part_probs[:, t]

            q = part_probs[:, t + 1]

            kl = (
                p * (
                    torch.log(p + 1e-8)
                    -
                    torch.log(q + 1e-8)
                )
            ).sum(dim=-1)

            kl_diff[:, t] = kl

        # ======================================================
        # Part Importance
        # ======================================================
        part_importance = torch.zeros(
            B,
            T,
            self.num_parts,
            device=part_feats.device
        )

        part_importance[:, 0] = kl_diff[:, 0]

        part_importance[:, 1:T] = kl_diff

        # ======================================================
        # Normalize
        # ======================================================
        min_imp = part_importance.amin(
            dim=(1, 2),
            keepdim=True
        )

        max_imp = part_importance.amax(
            dim=(1, 2),
            keepdim=True
        )

        part_importance = (
            part_importance - min_imp
        ) / (
            max_imp - min_imp + 1e-8
        )

        # ======================================================
        # Frame Weight
        # ======================================================
        frame_weights = part_importance.max(dim=2)[0]

        min_w = frame_weights.min(
            dim=1,
            keepdim=True
        )[0]

        max_w = frame_weights.max(
            dim=1,
            keepdim=True
        )[0]

        frame_weights = (
            frame_weights - min_w
        ) / (
            max_w - min_w + 1e-8
        )

        # ======================================================
        # Skeleton Fusion
        # ======================================================
        skel_feat = (
            part_feats
            *
            part_importance.unsqueeze(-1)
        ).sum(dim=2)

        # ======================================================
        # RGB + Skeleton Fusion
        # ======================================================
        fuse = torch.cat(
            [rgb_feat, skel_feat],
            dim=-1
        )

        fuse = torch.relu(
            self.fc_fuse(fuse)
        )

        # ======================================================
        # Position Encoding
        # ======================================================
        fuse = fuse + self.pos_encoder[:, :T]

        # ======================================================
        # Transformer Input
        # ======================================================
        fuse = fuse.permute(
            1,
            0,
            2
        )

        # (T, B, hidden_dim)

        # ======================================================
        # Attention Bias
        # ======================================================
        bias = (
            frame_weights.unsqueeze(1)
            +
            frame_weights.unsqueeze(2)
        )

        # (B, T, T)

        # ======================================================
        # Transformer
        # ======================================================
        for layer in self.transformer_layers:

            fuse = layer(
                fuse,
                bias=bias
            )

        # ======================================================
        # Recover Shape
        # ======================================================
        transformer_out = fuse.permute(
            1,
            0,
            2
        )

        # ======================================================
        # Temporal Pooling
        # ======================================================
        video_feat = transformer_out.mean(dim=1)

        # ======================================================
        # Classification
        # ======================================================
        logits = self.classifier(video_feat)

        return logits