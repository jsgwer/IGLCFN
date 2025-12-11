import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class IC9600Dataset(Dataset):

    def __init__(self, data_root, list_file_name, transform=None):
        super().__init__()
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'images')
        self.list_file_name = list_file_name
        self.transform = transform

        self.data_list = self._load_data_list()
        print(f"Successfully loaded {len(self.data_list)} samples from {self.list_file_name}.")

    def _load_data_list(self):
        data_list = []

        full_list_path = os.path.join(self.data_root, self.list_file_name)

        if not os.path.exists(full_list_path):
            raise FileNotFoundError(f"List file not found: {full_list_path}")

        with open(full_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.rsplit(None, 1)

                    if len(parts) != 2:
                        print(f"Warning: Skipping unparsable line: {line}")
                        continue

                    image_name, label_str = parts

                    image_path = os.path.join(self.image_dir, image_name)
                    label = float(label_str)

                    data_list.append((image_path, label))

                except Exception as e:
                    print(f"Error processing line '{line}': {e}")
                    continue

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label_float = self.data_list[idx]

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found, skipping sample: {image_path}")
            return None, None
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label_float, dtype=torch.float32)

        return image, label


class IC9600Dataset_heatmap(Dataset):

    def __init__(self, data_root, list_file_name, transform=None):
        super().__init__()
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, 'images')
        self.heatmap_dir = os.path.join(data_root, 'heatmap')

        self.list_file_name = list_file_name
        self.transform = transform

        if not os.path.exists(self.heatmap_dir):
            print(f"Warning: Heatmap directory not found: {self.heatmap_dir}")

        self.data_list = self._load_data_list()
        print(f"Successfully loaded {len(self.data_list)} samples from {self.list_file_name}.")

    def _load_data_list(self):
        data_list = []

        full_list_path = os.path.join(self.data_root, self.list_file_name)

        if not os.path.exists(full_list_path):
            raise FileNotFoundError(f"List file not found: {full_list_path}")

        with open(full_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.rsplit(None, 1)

                    if len(parts) != 2:
                        print(f"Warning: Skipping unparsable line: {line}")
                        continue

                    image_name, label_str = parts

                    image_path = os.path.join(self.image_dir, image_name)

                    base_name, _ = os.path.splitext(image_name)
                    heatmap_name = base_name + '.npy'
                    heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)

                    label = float(label_str)

                    data_list.append((image_path, heatmap_path, label))

                except Exception as e:
                    print(f"Error processing line '{line}': {e}")
                    continue

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, heatmap_path, label_float = self.data_list[idx]

        # 1. Read Image
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found, skipping sample: {image_path}")
            return None, None, None
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None, None, None

        # 2. Read and Process Heatmap
        try:
            heatmap_np = np.load(heatmap_path)

            min_val = np.min(heatmap_np)
            max_val = np.max(heatmap_np)
            if max_val > min_val:
                heatmap_np = (heatmap_np - min_val) / (max_val - min_val)
            else:
                heatmap_np = np.zeros_like(heatmap_np)

            heatmap_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8), mode='L')

            resize_transform = transforms.Resize((224, 224), interpolation=Image.BILINEAR)
            heatmap_resized_pil = resize_transform(heatmap_pil)

            heatmap_tensor = transforms.ToTensor()(heatmap_resized_pil)

        except FileNotFoundError:
            print(f"Error: Heatmap file not found, skipping sample: {heatmap_path}")
            return None, None, None
        except Exception as e:
            print(f"Error reading or processing heatmap {heatmap_path}: {e}")
            return None, None, None

        # 3. Image Preprocessing
        if self.transform:
            image = self.transform(image)

        # 4. Label Tensorization
        label = torch.tensor(label_float, dtype=torch.float32)

        return image, label, heatmap_tensor