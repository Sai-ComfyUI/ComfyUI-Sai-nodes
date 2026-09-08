# ComfyUI-Sai-nodes

這是一組使用 ComfyUI V3 API 的自訂節點，包含多參考圖 conditioning、FLUX.1 latent
放大、FFHQ 人臉對齊、MyTimeMachine 年齡轉換與個人化訓練，以及附標籤的圖片比較拼貼。

[English README](README.md)

## 安裝

將專案 clone 到 `ComfyUI/custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Sai-ComfyUI/ComfyUI-Sai-nodes.git
```

使用執行 ComfyUI 的同一套 Python 安裝依賴：

```bash
python -m pip install -r ComfyUI-Sai-nodes/requirements.txt
```

Windows portable 版可從安裝根目錄執行：

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Sai-nodes\requirements.txt
```

完成後重新啟動 ComfyUI。詳細步驟與疑難排解請看
[安裝與更新說明](docs/INSTALLATION.md)。

## 模型檔案

模型權重不會放進這個 Git repository：

- LUA-FLUX loader 可以從原作者的
  [`vaskers5/LUA-FLUX`](https://huggingface.co/vaskers5/LUA-FLUX) 自動下載。
- MyTimeMachine 模型請放在
  `ComfyUI-Sai-nodes/models/mytimemachine/`。文件會優先列出原專案或模型作者的下載來源；
  [`sailing/ComfyUI-Sai-nodes_Models`](https://huggingface.co/sailing/ComfyUI-Sai-nodes_Models/tree/main/mytimemachine)
  作為已驗證可下載的備援來源。

推論、人臉對齊、個人化推論及訓練所需的不同檔案，請看
[模型下載與目錄配置](docs/MODELS.md)。

## 功能與文件

- **Multi Reference Latent Ψ**：整理多張參考圖供 edit model 使用。
- **LUA FLUX**：載入官方模型，對 FLUX.1 latent 進行 x2／x4 放大。
- **MyTimeMachine**：FFHQ 人臉對齊與貼回、一般及個人化年齡轉換、個人化訓練。
- **Labeled Image Collage Ψ**：製作附標籤、可換行與儲存版型的比較拼貼。

完整節點清單與接法請看[使用說明](docs/USAGE.md)。個別節點的詳細輸入輸出文件位於
[`docs/`](docs/)；第三方程式碼與模型授權見
[Third-party notices](THIRD_PARTY_NOTICES.md)。
