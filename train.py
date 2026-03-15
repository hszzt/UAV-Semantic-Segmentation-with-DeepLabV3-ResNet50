# train.py 
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models.segmentation as seg
import matplotlib.pyplot as plt

# ==========================================
# 1. 函式 與 Definitions 
# ==========================================

# RLE 函式：壓縮 mask 成 RLE 格式
def rle_encode(mask):
    # flatten(order='F')：以欄為主展開 2D mask
    pixels = mask.flatten(order='F')    
    # 在頭尾補 0，方便找 transition
    pixels = np.concatenate([[0], pixels, [0]])
     # 找出值有改變的位置 (0->1 或 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 偶數位置改成 run length 
    runs[1::2] -= runs[:-1:2]
    # 把連續1的區段用起點 + 長度表示
    return ' '.join(str(x) for x in runs)

# Denormalize 函式：用於繪圖還原顏色
# --------------------------------------------------
# 用在錯誤分析畫圖時，將 tensor 還原回 [0,1] 的 RGB 方便顯示 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def denormalize(tensor):
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)    # x = x * std + mean
    img = tensor.numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 1)

# ==========================================
# 2. Transform 定義（影像 & mask 同步增強）
# ==========================================

class TrainTransform:
    """
    訓練時使用的 Transform：
    - 讓 img / mask 做同步的幾何變換（水平 / 垂直翻轉）
    - 顏色增強只作用在 img
    - 最後 ToTensor + Normalize
    """
    def __init__(self):
         # 顏色增強：亮度、對比、飽和度、色相
        self.color_aug = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def __call__(self, img, mask):
        """
        Parameters
            img : numpy array, shape (H, W, 3), RGB
            mask: numpy array, shape (H, W), class id（0~15）or None (test 時沒有 mask)

        Returns（PyTorch model 需要 tensor 格式）
            img : torch.Size(3, H, W) 的 float tensor（已 Normalize，符合 ResNet 的輸入規格）
            mask: torch.Size(H, W) 的 long tensor（給 CrossEntropyLoss 用）
        """

        # random horizontal flip
        if np.random.rand() < 0.5:
            # axis=1 => 左右翻轉
            img = np.ascontiguousarray(np.flip(img, axis=1))
            if mask is not None:
                # mask 也要同步翻轉
                mask = np.ascontiguousarray(np.flip(mask, axis=1))

        # random vertical flip
        if np.random.rand() < 0.5:
            # axis=1 => 左右翻轉
            img = np.ascontiguousarray(np.flip(img, axis=0))
            if mask is not None:
                mask = np.ascontiguousarray(np.flip(mask, axis=0))

        # color jitter （只作用在影像）
        img_pil = T.ToPILImage()(img)
        img_pil = self.color_aug(img_pil)
        img = np.array(img_pil)

        # ToTensor + Normalize
        img = T.ToTensor()(img) # 會把 img 從 HWC 轉成 CHW，並且除以 255
        img = T.Normalize(mean=MEAN, std=STD)(img)

        # mask 轉成 tensor.long 供 CrossEntropyLoss 使用
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask


class TestTransform:
    """
    驗證 / 測試用 Transform：
    - 不做隨機增強（結果穩定）
    - 只做 ToTensor + Normalize
    - mask 轉成 long tensor
    """
    def __call__(self, img, mask):
        img = T.ToTensor()(img)
        img = T.Normalize(mean=MEAN, std=STD)(img)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

# ==========================================
# 3. Dataset 定義
# ==========================================

class UAVDataset(Dataset):
    """
    用來讀取 UAV segmentation 資料集的 Dataset：
    - img_dir: 影像資料夾路徑
    - mask_dir: 標註資料夾路徑（若為 None，代表是 test set）
    - transform: 上面定義的 TrainTransform / TestTransform
    """
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 使用檔名排序，確保圖像與 mask 一一對應
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
         # 取得影像檔名與路徑
        filename = self.files[idx]
        img_path = os.path.join(self.img_dir, filename)

        # 讀取影像並轉成 RGB (cv2 預設是 BGR)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # original_size: 保留原始 H, W，之後 test 時要 resize 回去做 RLE
        original_size = img.shape[:2]  # (H, W)

        # 統一 resize 大小
        # target_size = (width, height) for cv2
        target_size = (512, 512)  # (W, H) 給 cv2 用
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 讀取對應的 mask（若有的話）
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, filename)
            # 單通道灰階，value = class id
            mask = cv2.imread(mask_path, 0) 
            # mask 用最近鄰插值，避免 class id 被內插成奇怪的小數
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # transform 同時處理 img & mask
        if self.transform:
            img, mask = self.transform(img, mask)
        else:
             # 沒給 transform 的 fallback
            img = T.ToTensor()(img)
            img = T.Normalize(mean=MEAN, std=STD)(img)
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.long)

        # train/val: 回傳 (img, mask)
        if self.mask_dir is not None:
            return img, mask
        # test: 回傳 (img, 檔名, 原始尺寸)，之後要還原成原圖大小再做 RLE
        else:
            return img, filename, original_size

# ==========================================
# 4. 模型建立與訓練 / 驗證函式
# ==========================================

def get_model(device):
    """
    建立 DeepLabV3-ResNet50 segmentation 模型：
    - 使用 torchvision 預訓練權重 (weights="DEFAULT")
    - 最後一層輸出 channel 改成 16（對應 16 個 class）
    """
    model = seg.deeplabv3_resnet50(weights="DEFAULT")
    # 原本 classifier[4] 是 Conv2d(256, 21, 1)，這裡改成 16 類
    model.classifier[4] = nn.Conv2d(256, 16, kernel_size=1)
    return model.to(device)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    單一 epoch 的訓練流程：
    - 走過一遍 train_loader
    - 前向傳遞 -> 計算 loss -> 反向傳遞 -> 更新參數
    - 回傳這個 epoch 的平均訓練 loss
    """
    model.train()
    total_loss = 0
    for step, (imgs, masks) in enumerate(loader):
        # 第一次 batch 印出 shape，確認資料尺寸有沒有問題
        if step == 0:
            print("  ▶ got first batch:", imgs.shape, masks.shape) 

        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)["out"]        # DeepLabV3 的輸出在 key "out"
        loss = loss_fn(outputs, masks)      # CrossEntropyLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 每 50 個 batch 印一次訓練進度 & 當下 loss
        if (step + 1) % 50 == 0:
            print(f"  [train] step {step+1}/{len(loader)}, loss={loss.item():.4f}")

    # 回傳平均 loss
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    """
    驗證流程：
    - 不做反向傳遞，只 forward 計算 loss
    - 回傳整個驗證集的平均 loss
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)['out']
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# ==========================================
# 5. 主程式：訓練 + 繪圖 + 產生 submission + 錯誤分析
# ==========================================
if __name__ == '__main__':

    # 啟用 cuDNN 的最佳化（會根據輸入尺寸自動選擇最佳實作）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # # ----------- 裝置設定 -----------
    if torch.cuda.is_available():
        device = torch.device("cuda")   # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # M4 用多台裝置測試
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # ----------- 路徑設定 (讀取 Dataset) -----------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "UAV_dataset")
    
    train_img_dir = os.path.join(DATA_DIR, "train", "imgs")
    train_mask_dir = os.path.join(DATA_DIR, "train", "masks")
    test_img_dir = os.path.join(DATA_DIR, "test", "imgs")
    
    # ----------- Transform 物件 -----------
    train_transform = TrainTransform()
    test_transform = TestTransform()

    # ----------- 建立 Dataset & Train/Val 切分 -----------
    full_dataset = UAVDataset(train_img_dir, train_mask_dir, transform=train_transform)
    
    # 選擇使用 80% 當訓練，20% 當驗證
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader 設定
    NUM_WORKERS = 4
    BATCH_SIZE = 8

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
     # ----------- 建立 model & optimizer & Scheduler & Loss -----------
    model = get_model(device)

    # Adam optimizer，學習率 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # StepLR：每 8 個 epoch 將 lr 乘上 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    
    # Class weights：
    # counts 是每個 class 的 pixel 數，為了處理 class imbalance，
    # 這裡使用 1/sqrt(count) 當作權重，再 normalize 讓總和為 1。
    counts = torch.tensor(
        [86780349, 62633229, 34434320, 22928416, 12488439, 12102789,
         8432305, 5246283, 4176087, 2762295, 2688490, 2140505,
         1533080, 1381492, 1248078, 1167843],
        dtype=torch.float
    )
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.sum()
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    # ----------- 訓練主迴圈 -----------
    EPOCHS = 30

    best_loss = float("inf")
    losses = []     # 紀錄每個 epoch 的 train loss（畫圖用）

    patience = 5    # early stopping：連續 5 個 epoch 沒進步就停止
    bad_epochs = 0

    print(f"Start Training for {EPOCHS} epochs...")

    history = []  # 用來輸出 train_log.csv
    
    for epoch in range(EPOCHS):
        # 一個 epoch 的訓練與驗證
        t_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        v_loss = evaluate(model, val_loader, loss_fn, device)
        losses.append(t_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | lr={current_lr:.6f}")

        # 紀錄 training log
        history.append({
            "epoch": epoch + 1,
            "train_loss": t_loss,
            "val_loss": v_loss,
            "lr": current_lr
        })

        # 更新 LR
        scheduler.step()

        # 儲存目前最好的模型（以 val_loss 判斷）
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  --> Saved Best Model (Val Loss: {v_loss:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            # 連續 patience 個 epoch 沒進步就 early stopping
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break
            
    print("Training Done. Best model saved.")

    # ----------- 儲存訓練 log 成 CSV -----------
    df_log = pd.DataFrame(history)
    os.makedirs("./log", exist_ok=True)
    df_log.to_csv("./log/train_log.csv", index=False)
    print("Epoch log saved to ./log/train_log.csv")

    # ==========================================
    # 6. 輸出訓練 Loss 曲線圖（作業要求：chart/figure）
    # ==========================================
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs('./fig', exist_ok=True)
    plt.savefig('./fig/training_loss_curve.png')
    print("Loss curve saved to ./fig/training_loss_curve.png")
    plt.close()

    # ==========================================
    # 7. Inference 階段：產生 submission.csv
    # ==========================================
    print("Start Generating Submission CSV...")
    
    # 載入最佳模型權重（以 val loss 最小為準）
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    # test dataset 不帶 mask_dir，transform 改用 TestTransform
    test_dataset = UAVDataset(test_img_dir, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    submission_rows = []
    
    with torch.no_grad():
        for i, (img, filename, original_size) in enumerate(test_loader):
            img = img.to(device)

            # TTA：對同一張圖做原圖 + 左右翻轉兩次推論，再平均
            output1 = model(img)['out']
            output2_flip = model(torch.flip(img, [3]))['out']
            output2 = torch.flip(output2_flip, [3])
            output_mean = (output1 + output2) / 2.0
            
            # 取每個 pixel 的最大機率 class
            pred_mask = torch.argmax(output_mean, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # 將預測 mask resize 回原始尺寸 (orig_w, orig_h)
            orig_h, orig_w = original_size[0].item(), original_size[1].item()
            pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # RLE
            row = {'img': filename[0]}
            for class_id in range(16):
                class_mask = (pred_mask == class_id).astype(np.uint8)
                if class_mask.sum() == 0:
                    row[f'class_{class_id}'] = 'none'
                else:
                    row[f'class_{class_id}'] = rle_encode(class_mask)
            submission_rows.append(row)

    # 組成 DataFrame 並輸出 submission.csv  
    df_submit = pd.DataFrame(submission_rows)
    cols = ['img'] + [f'class_{i}' for i in range(16)]
    df_submit = df_submit[cols]
    df_submit.to_csv("submission.csv", index=False)
    print("Submission saved to 'submission.csv'.")

    # ==========================================
    # 8. Error analysis visualization
    # ==========================================
    print("\nStarting Error Analysis Visualization...")

    # 重新建立一個用 TestTransform 的 train_dataset_eval
    # 目的是用不做隨機增強的版本，看模型在原圖上的錯誤情況
    train_dataset_eval = UAVDataset(train_img_dir, train_mask_dir, transform=test_transform)
    num_examples = 3
    current_count = 0
    
    # 隨機抽樣 20 張訓練影像，從中挑錯誤率較高的畫出來
    indices = np.random.choice(len(train_dataset_eval), 20, replace=False)

    for idx in indices:
        if current_count >= num_examples:
            break
            
        img_tensor, mask_true = train_dataset_eval[idx]
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # error_map = 1 表示 prediction 與 ground truth 不同（錯誤）
        mask_true_np = mask_true.numpy()
        error_map = (pred_mask != mask_true_np).astype(np.uint8)
        error_rate = np.sum(error_map) / error_map.size
        
        # 若錯誤率太低就跳過
        if error_rate < 0.05:
            continue

        # 繪製：原圖 / GT / prediction / error map
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(denormalize(img_tensor))
        axes[0].set_title(f"Image {idx} (RGB)")
        axes[1].imshow(mask_true_np, cmap='tab20', vmin=0, vmax=15)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=15)
        axes[2].set_title("Model Prediction")
        axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title(f"Error Map (Red=Wrong)\nError Rate: {error_rate:.2%}")
        
        for ax in axes:
            ax.axis('off')
        
        save_path = f'./fig/error_analysis_{idx}.png'
        plt.savefig(save_path)
        print(f"  -> Error analysis saved: {save_path}")
        plt.close(fig)
        
        current_count += 1

    if current_count == 0:
        print("Note: Could not find images with >5% error rate for analysis.")
    
    print("\n--- All tasks completed successfully. ---")
