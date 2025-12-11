import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple
from Model.LKA import LargeKernelAttention
import os


class ConvNeXtToViTCrossAttention(nn.Module):

    def __init__(self,
                 convnext_dim: int = 640,
                 vit_dim: int = 192,
                 head_num: int = 8,
                 feat_size: Tuple[int, int] = (7, 7),
                 attn_dim: int = 256):
        super().__init__()

        self.H, self.W = feat_size
        self.patch_num = self.H * self.W
        self.vit_dim = vit_dim
        self.key_proj = nn.Linear(convnext_dim, attn_dim)
        self.value_proj = nn.Linear(convnext_dim, attn_dim)

        self.query_proj = nn.Linear(vit_dim, attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=head_num,
            batch_first=False
        )

        self.output_proj = nn.Linear(attn_dim, vit_dim)

        self.norm = nn.LayerNorm(vit_dim)

    def forward(self, convnext_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        N = vit_feat.shape[0]

        convnext_sequence = convnext_feat.flatten(2).transpose(1, 2)
        K_src = convnext_sequence
        V_src = convnext_sequence

        vit_sequence = vit_feat.flatten(2).transpose(1, 2)
        Q_tgt = vit_sequence

        K_src_T = K_src.transpose(0, 1)
        V_src_T = V_src.transpose(0, 1)
        Q_tgt_T = Q_tgt.transpose(0, 1)

        query = self.query_proj(Q_tgt_T)
        key = self.key_proj(K_src_T)
        value = self.value_proj(V_src_T)

        attn_output, _ = self.cross_attn(query, key, value)

        attn_output = self.output_proj(attn_output)

        fused_sequence = self.norm(attn_output + Q_tgt_T)

        fused_feat = fused_sequence.transpose(0, 1)
        fused_feat = fused_feat.transpose(1, 2)
        fused_feat = fused_feat.reshape(N, self.vit_dim, self.H, self.W)

        return fused_feat + vit_feat


class FeatureSelfAttention(nn.Module):

    def __init__(self, in_dim=384, num_heads=8, dropout=0.1, mlp_expand=2):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == in_dim, "Input channel must be divisible by number of attention heads"

        self.pre_norm = nn.LayerNorm(in_dim)

        self.qkv_proj = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.out_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.residual_drop = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * mlp_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * mlp_expand, in_dim)
        )

    def _flatten_2d_to_seq(self, feat):
        N, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(N, H * W, C)

    def _restore_seq_to_2d(self, seq, H=7, W=7):
        N, seq_len, C = seq.shape
        return seq.reshape(N, H, W, C).permute(0, 3, 1, 2)

    def self_attention_block(self, feat_seq):
        N, seq_len, C = feat_seq.shape

        qkv = self.qkv_proj(feat_seq)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(N, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_score = torch.matmul(q, k)
        attn_score = attn_score / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=attn_score.device))
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.attn_drop(attn_weight)

        attn_out = torch.matmul(attn_weight, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(N, seq_len, C)
        attn_out = self.out_proj(attn_out)
        attn_out = self.residual_drop(attn_out)

        return feat_seq + attn_out

    def forward(self, feat):
        feat_seq = self._flatten_2d_to_seq(feat)

        normed_feat = self.pre_norm(feat_seq)
        attn_feat = self.self_attention_block(normed_feat)

        normed_attn = self.mlp_norm(attn_feat)
        refined_feat = self.mlp(normed_attn)
        refined_feat = self.residual_drop(refined_feat)

        final_seq = attn_feat + refined_feat

        final_feat = self._restore_seq_to_2d(final_seq)

        return final_feat


class Fusion(nn.Module):
    def __init__(self, vit_channel=192, convnext_channel=640):
        super(Fusion, self).__init__()
        self.localView = nn.Sequential(
            nn.Conv2d(vit_channel * 2, vit_channel, 3, 1, 1),
            nn.BatchNorm2d(vit_channel),
            nn.ReLU(inplace=True)
        )
        self.alin = nn.Sequential(
            nn.Conv2d(convnext_channel, vit_channel, 1),
            nn.BatchNorm2d(vit_channel),
            nn.ReLU(inplace=True)
        )
        self.globalView = ConvNeXtToViTCrossAttention(vit_dim=vit_channel, convnext_dim=convnext_channel)
        self.finalFusion = FeatureSelfAttention(vit_channel * 2)

    def forward(self, vit_feat, conv_feat):
        vit_feat = self.globalView(vit_feat=vit_feat, convnext_feat=conv_feat)
        conv_feat = self.alin(conv_feat)
        conv_feat = self.localView(torch.cat((conv_feat, vit_feat), dim=1)) + conv_feat
        final_feat = self.finalFusion(torch.cat((conv_feat, vit_feat), dim=1))
        return final_feat


class IGLCFN(nn.Module):
    def __init__(self, freeze_flag=False):
        super(IGLCFN, self).__init__()
        vit_tiny = timm.create_model('vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k', pretrained=True,
                                     num_classes=1)
        convnext_nano = timm.create_model('convnext_nano.d1h_in1k', pretrained=True,
                                          num_classes=1)
        self.freeze_flag = freeze_flag
        self.vit_backbone = nn.Sequential(*list(vit_tiny.children())[0:8])
        self.convnext_stage1 = nn.Sequential(
            convnext_nano.stem,
            convnext_nano.stages[0]
        )
        self.convnext_stage2 = convnext_nano.stages[1]
        self.convnext_stage3 = convnext_nano.stages[2]
        self.convnext_stage4 = convnext_nano.stages[3]

        self.lka3 = LargeKernelAttention(320)
        self.lka4 = LargeKernelAttention(640)
        if self.freeze_flag:
            self.genheat = nn.Sequential(
                nn.Conv2d(384, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
            self.imgFusion = nn.Sequential(
                nn.Conv2d(384 * 2, 384, 3, 1, 1),
                nn.BatchNorm2d(384),
                nn.GELU()
            )
            self.expand = nn.Sequential(
                nn.Conv2d(3, 80, 3, 2, 1),
                nn.BatchNorm2d(80),
                nn.GELU(),
                nn.Conv2d(80, 384, 3, 1, 1),
                nn.BatchNorm2d(384),
                nn.GELU()
            )
        self.fusion = Fusion(192, 640)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        feature_img = feature
        feature_vit = self.vit_backbone(feature)
        feature_vit = feature_vit.transpose(1, 2).reshape(-1, 192, 7, 7)
        feature_convnext = self.convnext_stage1(feature)
        feature_convnext = self.convnext_stage2(feature_convnext)
        feature_convnext = self.convnext_stage3(feature_convnext)
        feature_convnext = self.lka3(feature_convnext) + feature_convnext
        feature_convnext = self.convnext_stage4(feature_convnext)
        feature_convnext = self.lka4(feature_convnext) + feature_convnext
        feature = self.fusion(feature_vit, feature_convnext)
        if self.freeze_flag:
            feature_img = self.expand(feature_img)
            feature_scale = F.interpolate(feature, size=(feature_img.shape[-2], feature_img.shape[-1]), mode="bilinear",
                                          align_corners=True)
            feature_img = self.imgFusion(torch.cat((feature_img, feature_scale), dim=1))
            heatmap = self.genheat(feature_img)

        feature = self.avgPool(feature).squeeze()
        score = self.head(feature).squeeze()
        if self.freeze_flag:
            return score, heatmap
        else:
            return score


def main(weights_path: str = 'iglc_best.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = IGLCFN().to(device)
    except Exception as e:
        print(f"Model instantiation failed, check timm dependency or LargeKernelAttention implementation: {e}")
        return

    if os.path.exists(weights_path):
        print(f"Loading weights file: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Weights loading failed, check if model structure and weights file match: {e}")
            return
    else:
        print(f"Warning: Weights file {weights_path} not found. Using untrained model for testing.")

    model.eval()
    print("Model set to evaluation mode (model.eval()).")

    N, C, H, W = 1, 3, 224, 224
    dummy_input = torch.randn(N, C, H, W).to(device)
    print(f"\nPreparing test input data: Shape {dummy_input.shape}")

    print("Starting forward pass prediction...")
    with torch.no_grad():
        score, heatmap = model(dummy_input)

    print("--- Prediction Results ---")
    print(f"Predicted Score: {score.item():.4f}")
    print(f"Heatmap Output Shape: {heatmap.shape}")
    print(f"Heatmap Max Value: {heatmap.max().item():.4f}")
    print(f"Heatmap Min Value: {heatmap.min().item():.4f}")
    print("----------------")

    if 0 <= score.item() <= 1 and 0 <= heatmap.min().item() and heatmap.max().item() <= 1:
        print("✅ Prediction output conforms to the [0, 1] range of the Sigmoid activation function.")
    else:
        print("❌ Warning: Prediction output may be outside the [0, 1] range.")


if __name__ == '__main__':
    main(weights_path=r'.\checkpoints\IGLCFN.pth')