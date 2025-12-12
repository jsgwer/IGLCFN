\# Based on interpretable perception-driven image complexity prediction and heatmap generation algorithm
This repository provides the official code implementation for the paper **"Interpretable Perception-Driven Image Complexity Prediction and Heatmap Generation"** (基于可解释的感知驱动图像复杂度预测及热力图生成算法). 

The code realizes image complexity prediction with interpretability and heatmap generation, which can effectively reflect the key regions contributing to image complexity perception.


## 📊 Dataset Preparation
The model is trained and tested on the IC9600 dataset. Please follow the steps below to prepare the dataset:
1. Refer to the IC9600 repository for the dataset acquisition method: https://github.com/tinglyfeng/IC9600
2. Place the acquired IC9600 dataset in the root directory of this project, ensuring the path is:
   \'
   ./IC9600
   \'
   The dataset structure should include training set, test set, and corresponding annotation files (consistent with the original IC9600 repository).

## 🚀 Model Training
To train the image complexity prediction and heatmap generation model from scratch:
1. Ensure the IC9600 dataset is correctly placed in './IC9600'.
2. Run the training script:
   '
   python train.py
   '
3. The trained model checkpoints will be saved in the './cks' directory (automatically created if not exists). 

### Training Configuration
You can adjust hyperparameters (e.g., learning rate, batch size, epoch number, model backbone) by modifying the parameters in \`options.py\` or adding command-line arguments according to your hardware conditions and training needs.

## 🧪 Model Testing & Inference
### Use Pre-trained Model
We provide a pre-trained model for quick testing. Follow these steps:
1. Download the pre-trained model from Baidu Netdisk:
   - Link: https://pan.baidu.com/s/1u8Pn46lxhbuqQPP0vxLG3A
   - Extraction code: y5r3
2. Place the downloaded model checkpoint file into the './checkpoints' directory.

### Run Test Script
Execute the test script to predict the IC9600 test set and generate complexity heatmaps:
'
python test.py
'
The test script will:
- Load the pre-trained/trained model parameters.
- Evaluate the model performance on the IC9600 test set (output metrics such as MAE, RMSE for complexity prediction).
- Generate interpretable heatmaps for test set images, saving them to the specified directory
