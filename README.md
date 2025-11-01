# 🚗 Multi-Head Car Image Classifier  
**Swin Transformer + EfficientNet | Brand • Model • Year Prediction with Unknown Detection**

This project trains a **multi-task deep learning model** that predicts:

- ✅ Car **brand**
- ✅ Car **model**
- ✅ Car **manufacturing year**

It uses two backbones:

| Backbone | Framework | Purpose |
|---------|-----------|--------|
| Swin-Tiny | PyTorch + timm | Main multi-head classifier |
| EfficientNetB0 | TensorFlow/Keras | Alternative CNN backbone |

Includes **confidence-based Unknown filtering**, where low-confidence predictions are replaced with `-1`.

---

## 📂 Dataset

The dataset used for training & inference can be downloaded from Kaggle:

🔗 Kaggle Dataset for training: 
> https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset/data 

Kaggle Dataset for training:
> https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset


### Expected CSV format


### For Colab users
```python
from google.colab import drive
drive.mount("/content/drive")

DATA_DIR = "/content/drive/MyDrive/datasets/cars/"

Model Architecture (Multi-Head) : 

[Swin / EfficientNet Feature Extractor]
               ↓
      Global Pooling
               ↓
 ┌─────────┬──────────┬────────┐
 | Brand   | Model    | Year   |   ← separate Dense layers
 └─────────┴──────────┴────────┘

Loss = sum of three cross-entropy losses (with label smoothing & class weights for year)

Unknown mode:
If softmax_conf < threshold, output = -1
------------------------------------------------------------------------------------
Training
Swin Transformer (PyTorch)

Run notebook:

notebooks/swin_multihead.ipynb

Training strategy:

Stage 1: freeze Swin backbone, train heads

Stage 2: unfreeze last layers, fine-tune with smaller LR

Model checkpoint saved as: 

checkpoints/swin_multihead.pth

EfficientNet (TensorFlow)

Notebook:

notebooks/efficientnet_multihead.ipynb
------------------------------------------------------------------------
Training strategy:

Freeze backbone → train heads

Then unfreeze final 20 layers → fine tune

Model saved as:

auto_image.keras


Inference
Swin (PyTorch)

Set paths inside notebook/script:

TEST_DIR = "/content/cars_test"
WEIGHTS_PATH = "/content/drive/MyDrive/checkpoints/swin/swin_multihead.pth"
OUT_CSV = "swin_test_preds.csv"

TAU = {"brand":0.50, "model":0.50, "year":0.50}  # confidence threshold

EfficientNet (Keras) : 

model = tf.keras.models.load_model("auto_image.keras")

| image     | brand  | conf | model  | conf | year   | conf |
| --------- | ------ | ---- | ------ | ---- | ------ | ---- |
| 00001.jpg | 3      | 0.86 | 2      | 0.81 | 5      | 0.70 |
| 00002.jpg | **-1** | 0.54 | **-1** | 0.50 | **-1** | 0.41 |


-1 = Unknown (model not confident enough)

A tiny sample result CSV is included under outputs/.
------------------------------------------------------------------
Installation : 

git clone https://github.com/<your-username>/car-multihead-classifier.git
cd car-multihead-classifier
pip install -r requirements.txt

-------------------------------------------------------------------

Requirements : 

See requirements.txt — includes:

.PyTorch + timm

.TensorFlow / Keras

.Pandas, Pillow, NumPy, SkLearn, Matplotlib

---------------------------------------------------------------------

```markdown
## 📥 Download Trained Weights

Due to GitHub file size limits, model checkpoints are provided via Google Drive:

| Model | File | Download Link |
|------|------|---------------|
EfficientNetB0 | `auto_image.keras` | https://drive.google.com/file/d/1eat3O5ZCp7RRg_rJeL4sKuPAyNakXmIU/view?usp=sharing

Swin Transformer | `swin_multihead_best.pth` | https://drive.google.com/file/d/1fEamCJvvPS7_6echIWpYKI6BfwHqiIqX/view?usp=sharing

------------------------------------------------------------------

Folder Structure :

car-multihead-classifier/
│
├─ notebooks/                   # Jupyter notebooks for both backbones
├─ src/                         
├─ checkpoints/                 
├─ outputs/                     # sample CSV output
├─ requirements.txt
└─ README.md

---------------------------------------------------------------------

Results : 

CSV format file outputs including :

.Multi-task learning for automotive classification

.Swin Transformer achieves strong brand/model accuracy but for year class , it was not successful.

.Unknown gating avoids wrong confident predictions

--------------------------------------------------------------------

👤 Author
Amin Salehi Tabrizi (@aminmech)
📍 Machine Learning Engineer & Computational Scientist
📫 github.com/aminmech
Email : tabrizi15@itu.edu.tr

---------------------------------------------------------------------

Credits :

.Swin Transformer

.EfficientNet

.timm library


https://drive.google.com/file/d/1eat3O5ZCp7RRg_rJeL4sKuPAyNakXmIU/view?usp=sharing

https://drive.google.com/file/d/1fEamCJvvPS7_6echIWpYKI6BfwHqiIqX/view?usp=sharing