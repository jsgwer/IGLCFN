import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from options import args
from freeze import freeze_by_names
from Model.IGLCFN import IGLCFN
from dataset import IC9600Dataset, IC9600Dataset_heatmap
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaInfo(score, label):
    score = np.array(score)
    label = np.array(label)

    RMAE = np.sqrt(np.abs(score - label).mean())
    RMSE = np.sqrt(np.mean(np.abs(score - label) ** 2))
    Pearson = np.float64(pearsonr(label, score)[0])
    Spearmanr = np.float64(spearmanr(label, score)[0])
    tx = score
    ty = label
    ccc = (2 * Pearson * tx.std() * ty.std()) / (np.power(tx.mean() - ty.mean(), 2) + tx.var() + ty.var())

    info = ' RMSE : {:.4f} ,   RMAE : {:.4f} ,   Pearsonr : {:.4f} ,   Spearmanr : {:.4f} ,   CCC : {:.4f}'.format(
        RMSE, RMAE, Pearson, Spearmanr, ccc)
    print(info)

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, device, freeze_flag):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_list = []

        self.best_loss = float('inf')
        self.freeze_flag = freeze_flag

        if self.freeze_flag:
            self.loss_function_score = nn.MSELoss('mean')
        else:
            self.loss_function_score = nn.MSELoss()

        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, betas=(0.9, 0.999),
                      eps=1e-08,
                      weight_decay=0.01,
                      amsgrad=False)

        self.iter_per_epoch = len(self.train_loader)
        if args.warm > 0:
            self.warmup_scheduler = WarmUpLR(self.optimizer, self.iter_per_epoch * args.warm)
        else:
            self.warmup_scheduler = None

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epoch - args.warm,
            eta_min=args.lr * 1e-2
        )

    def train_epoch(self, epoch):
        running_loss = 0.0
        self.model.train()

        if self.freeze_flag:
            for batch_index, (image, _, heatmap) in enumerate(self.train_loader):
                image = image.to(self.device)
                heatmap = heatmap.to(self.device)

                self.optimizer.zero_grad()
                score_final, complex_map = self.model(image)
                complex_map = F.interpolate(complex_map, size=(heatmap.shape[-2], heatmap.shape[-1]),
                                              mode="bilinear",
                                              align_corners=True)
                loss = self.loss_function_score(heatmap, complex_map)

                loss.backward()
                self.optimizer.step()

                if self.warmup_scheduler and epoch <= args.warm:
                    self.warmup_scheduler.step()

                running_loss += loss.item()

                if (batch_index + 1) % (len(self.train_loader) // 3) == 0:
                    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}\tLR: {:0.6f}'.format(
                        loss.item(),
                        self.optimizer.param_groups[0]['lr'],
                        epoch=epoch,
                        trained_samples=batch_index * args.batch_size + len(image),
                        total_samples=len(self.train_loader.dataset)
                    ))
        else:
            for batch_index, (image, label) in enumerate(self.train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                score_final = self.model(image)
                loss = self.loss_function_score(score_final, label)

                loss.backward()
                self.optimizer.step()

                if self.warmup_scheduler and epoch <= args.warm:
                    self.warmup_scheduler.step()

                running_loss += loss.item()

                if (batch_index + 1) % (len(self.train_loader) // 3) == 0:
                    print(
                        'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tloss: {:0.4f}\tLR: {:0.6f}'.format(
                            loss.item(),
                            self.optimizer.param_groups[0]['lr'],
                            epoch=epoch,
                            trained_samples=batch_index * args.batch_size + len(image),
                            total_samples=len(self.train_loader.dataset)
                        ))

        avg_epoch_loss = running_loss / len(self.train_loader)
        self.loss_list.append(avg_epoch_loss)
        self.plot_loss_curve()
        return avg_epoch_loss

    def evaluation(self):
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            if self.freeze_flag:
                for image, label, heatmap in self.test_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    score_final, complex_map = self.model(image)

                    all_scores.extend(score_final.tolist())
                    all_labels.extend(label.tolist())
            else:
                for image, label in self.test_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    score_final = self.model(image)

                    all_scores.extend(score_final.tolist())
                    all_labels.extend(label.tolist())
        evaInfo(score=all_scores, label=all_labels)

    def run(self):
        print("--- Training Started ---")
        for epoch in range(1, args.epoch + 1):

            current_loss = self.train_epoch(epoch)

            self.evaluation()

            if current_loss < self.best_loss:
                torch.save(self.model.state_dict(), os.path.join(args.ck_save_dir, 'ck_best'))
                self.best_loss = current_loss
                print(f"✅ Epoch {epoch}: Loss {current_loss:.4f} is better than {self.best_loss:.4f}, saved best weights.")

            if epoch > args.warm:
                self.scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(args.ck_save_dir, f'IGLCFN{epoch}.pth'))
        print("--- Training Finished ---")

    def plot_loss_curve(self):
        iterations = range(1, len(self.loss_list) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.loss_list, label='Loss over Iterations')
        plt.title('Loss Curve')
        plt.xlabel('Iterations (Epochs)')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("plt.png", dpi=300)


def setup_dataloaders(freeze_flag):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if freeze_flag:
        train_dataset = IC9600Dataset_heatmap(
            data_root="./IC9600/",
            list_file_name="train.txt",
            transform=train_transform
        )
    else:
        train_dataset = IC9600Dataset(
            data_root="./IC9600/",
            list_file_name="train.txt",
            transform=train_transform
        )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True
                                  )

    if freeze_flag:
        test_dataset = IC9600Dataset_heatmap(
            data_root="./IC9600/",
            list_file_name="test.txt",
            transform=test_transform
        )
    else:
        test_dataset = IC9600Dataset(
            data_root="./IC9600/",
            list_file_name="test.txt",
            transform=test_transform
        )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False
                                 )
    return train_dataloader, test_dataloader


def setup_model_and_device(model_class, flag):
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    model = model_class(flag)
    model.to(device)

    if flag:
        model.load_state_dict(torch.load('./checkpoints/IGLCFN.pth', map_location=torch.device(device)),
                              strict=False)
        freeze_by_names(model, (
            'vit_backbone', 'convnext_stage1', 'convnext_stage2', 'convnext_stage3', 'convnext_stage4', 'lka3', 'lka4',
            'fusion', 'avgPool', 'head'))

    print("--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("------------------")

    return model, device


if __name__ == "__main__":

    if not os.path.exists(args.ck_save_dir):
        os.mkdir(args.ck_save_dir)

    model_instance, device_instance = setup_model_and_device(IGLCFN, args.freeze_flag)
    train_dl, test_dl = setup_dataloaders(args.freeze_flag)

    trainer = ModelTrainer(
        model=model_instance,
        train_loader=train_dl,
        test_loader=test_dl,
        device=device_instance,
        freeze_flag = args.freeze_flag
    )

    trainer.run()