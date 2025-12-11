import argparse
import os
from typing import Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
from Model.IGLCFN import IGLCFN
from torchvision import transforms

parser = argparse.ArgumentParser(description="Sliding Window Inference for Heatmap Generation")
parser.add_argument('-it', '--input_txt', type=str, default='./example/example.txt', help="Input TXT list path.")
parser.add_argument('-im', '--input_img', type=str, default='./example/images', help="Input images directory.")
parser.add_argument('-o', '--output', type=str, default='./out',
                    help="Output directory for heatmaps and blended images.")
parser.add_argument('-d', '--device', type=int, default=0, help="GPU device ID.")
parser.add_argument('-w', '--weights', type=str, default='IGLCFN.pth',
                    help="Model weights filename in checkpoint folder.")
parser.add_argument('--modelName', type=str, default='IGLCFN', help="Name of the model to load.")
parser.add_argument('--window_size', type=int, default=32, help="Size of the sliding window (HxW).")
parser.add_argument('--stride', type=int, default=2, help="Stride of the sliding window.")
parser.add_argument('--batch_size', type=int, default=97, help="Batch size for patch inference.")

def get_inference_transform(target_size: int = 224):
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def blend(ori_img_np: np.ndarray, ic_img_np: np.ndarray, alpha: float = 0.8, cm=plt.get_cmap("magma")) -> np.ndarray:
    cm_ic_map = cm(ic_img_np / 255.0)

    heatmap_rgb_255 = (cm_ic_map[:, :, :3] * 255).astype(np.uint8)

    heatmap = Image.fromarray(heatmap_rgb_255)
    ori_img = Image.fromarray(ori_img_np)

    if ori_img.size != heatmap.size:
        heatmap = heatmap.resize(ori_img.size)

    blend_img = Image.blend(ori_img, heatmap, alpha=alpha)
    return np.array(blend_img)

class InferenceEngine:
    def __init__(self, model, device: torch.device, window_size: int, stride: int, batch_size: int):
        self.model = model
        self.device = device
        self.window_size = (window_size, window_size)
        self.stride = stride
        self.batch_size = batch_size
        self.inference_transform = get_inference_transform(target_size=224)
        self.model.eval()

    def predict_with_sliding_window(self, global_score: float, img_filename: str) -> None:
        w_size, h_size = self.window_size
        global_score_tensor = torch.tensor(global_score, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            img_path = os.path.join(args.input_img, img_filename)
            ori_img = Image.open(img_path).convert("RGB")
            ori_img_np = np.array(ori_img)
            ori_height, ori_width = ori_img.height, ori_img.width

            image = self.inference_transform(ori_img).to(self.device).unsqueeze(0)

            N, C, H, W = image.shape

            patches_hw = image.unfold(2, h_size, self.stride).unfold(3, w_size, self.stride)
            patches = patches_hw.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, h_size, w_size)

            if patches.shape[0] == 0:
                print(f"Warning: No patches generated for {img_filename}.")
                return

            i_coords = torch.arange(0, H - h_size + 1, self.stride, device=self.device)
            j_coords = torch.arange(0, W - w_size + 1, self.stride, device=self.device)
            grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing='ij')

            flat_i_coords = grid_i.flatten()
            flat_j_coords = grid_j.flatten()

            all_predictions = []
            num_patches = patches.shape[0]
            for i in range(0, num_patches, self.batch_size):
                batch_patches = patches[i:i + self.batch_size]

                batch_patches_resized = F.interpolate(
                    batch_patches, size=(224, 224), mode="bilinear", align_corners=True
                )

                predictions = self.model(batch_patches_resized)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                all_predictions.append(predictions.squeeze())

            all_predictions = torch.cat(all_predictions)

            result_map = torch.zeros(H * W, device=self.device)
            count_map = torch.zeros(H * W, device=self.device)

            patch_i_indices = torch.arange(h_size, device=self.device).unsqueeze(0)
            patch_j_indices = torch.arange(w_size, device=self.device).unsqueeze(0)
            patch_indices_h, patch_indices_w = torch.meshgrid(patch_i_indices.squeeze(), patch_j_indices.squeeze(),
                                                              indexing='ij')

            patch_indices_h = patch_indices_h.flatten()
            patch_indices_w = patch_indices_w.flatten()

            global_i_indices = flat_i_coords.unsqueeze(1) + patch_indices_h
            global_j_indices = flat_j_coords.unsqueeze(1) + patch_indices_w
            global_flat_indices = global_i_indices * W + global_j_indices

            predictions_expanded = all_predictions.unsqueeze(1).expand(-1, h_size * w_size)

            result_map.scatter_add_(0, global_flat_indices.flatten(), predictions_expanded.flatten())

            ones = torch.ones_like(predictions_expanded)
            count_map.scatter_add_(0, global_flat_indices.flatten(), ones.flatten())

            result_map = result_map.view(H, W)
            count_map = count_map.view(H, W)

            count_map[count_map == 0] = 1
            final_result = result_map / count_map

            mean_final_result = torch.mean(final_result)
            if mean_final_result > 1e-8:
                final_result_normalized = final_result * (global_score_tensor / mean_final_result)
            else:
                final_result_normalized = torch.zeros_like(final_result)

            final_result_normalized[final_result_normalized > 1] = 1

            final_result_cpu = final_result_normalized.cpu()
            ic_map = F.interpolate(
                final_result_cpu.unsqueeze(0).unsqueeze(0),
                (ori_height, ori_width),
                mode='bilinear'
            ).squeeze()

            ic_map_img_np = (ic_map.numpy() * 255).round().astype('uint8')
            img_name_base = img_filename.rsplit('.', 1)[0]

            out_img_npy_name = f"{img_name_base}.npy"
            out_img_path_npy = os.path.join(args.output, out_img_npy_name)
            os.makedirs(os.path.dirname(out_img_path_npy), exist_ok=True)
            np.save(out_img_path_npy, ic_map_img_np)
            print(f"Processed: {img_filename}, Saved Heatmap NPY to: {out_img_path_npy}")

            blend_img = blend(ori_img_np, ic_map_img_np)
            out_img_png_name = f"{img_name_base}_blend.png"
            out_img_path_png = os.path.join(args.output, out_img_png_name)

            cv2.imwrite(out_img_path_png, blend_img[:, :, ::-1])
            print(f"Saved Blended PNG to: {out_img_path_png}")

def infer_txt(txt_path: str, engine: InferenceEngine):
    if not os.path.exists(txt_path):
        print(f"Error: List file not found: {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing images"):
        last_space_idx = line.rfind(' ')
        if last_space_idx == -1:
            print(f"Warning: Skipping line with incorrect format: {line}")
            continue

        try:
            img_filename = line[:last_space_idx].strip()
            value_str = line[last_space_idx + 1:].strip()
            score = float(value_str)

            engine.predict_with_sliding_window(score, img_filename)

        except Exception as e:
            print(f"Error processing line '{line}': {e}")

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("Error: The 'cv2' library (OpenCV-Python) is required for saving the blended PNG images.")
        exit()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Using device: {device}")

    try:
        model = IGLCFN().to(device)
    except TypeError:
        model = IGLCFN(True).to(device)

    weights_path = os.path.join('./checkpoints/', args.weights)

    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location=device),
            strict=False
        )
        print(f"Model weights loaded successfully from: {weights_path}")
    else:
        print(f"Warning: Model weights not found at {weights_path}. Using random initialization or default weights.")

    engine = InferenceEngine(
        model=model,
        device=device,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size
    )

    infer_txt(args.input_txt, engine)