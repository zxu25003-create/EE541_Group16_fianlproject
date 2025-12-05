## Environment Setup

This project is developed with **Python + PyTorch**.  
We recommend using **conda** to manage the environment.

### 1. Create and activate conda environment

conda create -n urbansound python=3.10 -y
conda activate urbansound

If you already created the environment before, just run:
conda activate urbansound
### 2. Install dependencies
GPU version (recommended, CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install librosa soundfile numpy scipy pandas scikit-learn matplotlib tqdm

### 3.run models
## model a and c
this repository contains the code for our UrbanSound8K experiments focusing on Model A and Model C:
Model A – VGG13 on log-mel spectrograms, trained from scratch with both time-domain and spectrogram-domain data augmentation.
Model C – VGG13 on log-mel spectrograms, trained from scratch without any data augmentation (baseline).
Both models share the same preprocessing, architecture, and 10-fold cross-validation protocol. The only difference is whether augmentation is enabled.
# 1.Project structure
The relevant directory layout (from this project) is:
EE541_Group16_fianlproject/
└── modela_c/
    ├──checkpoints/        # saved model checkpoints
    ├── data/
    │   └── UrbanSound8K/
    │       ├── audio/          # fold1, fold2, ..., fold10 subfolders
    │       └── metadata/
    │           └── UrbanSound8K.csv
    └── src/
        ├── augment.py          # time-domain augmentation (WaveformAugment)
        ├── specAugment.py      # spectrogram-domain augmentation (SpecAugment)
        ├── dataset.py          # UrbanSoundMelDataset
        ├── models.py           # VGG13Mel model definition
        ├── train.py            # training script for models A and C (and others)
        ├── eval_ac_all_folds.py# evaluation + visualization for A and C
        └── ...                 # (optional) run_all_folds.py, etc.
From now on we assume that all commands are run inside modela_c/src.
# 2.Dataset setup
1.Download UrbanSound8K from the official website.
2.Unzip it and place the folder under code/data/ so that you have:
modela_c/data/UrbanSound8K/
    audio/fold1/*.wav
    ...
    audio/fold10/*.wav
    metadata/UrbanSound8K.csv
3.The default paths in train.py are:
--csv_path        ../data/UrbanSound8K/metadata/UrbanSound8K.csv
--audio_base_path ../data/UrbanSound8K/audio
If your directory structure is different, you can override these arguments on the command line.
# 3.Training
The main training script is train.py.
- 3.1 Fold splitting
We follow the official 10-fold split. In train.py, build_folds(test_fold) does:
For the given test_fold (1–10):
test_folds = [test_fold]
From the remaining 9 folds:
The largest index is used as val_folds
The other 8 folds are used as train_folds
So for --fold 1:
Train: folds 2–9
Val: fold 10
Test: fold 1
We repeat this for all 10 folds.
- 3.2 training a single fold
From /src
cd model/src

- Train Model C (baseline, no augmentation) on fold 1
python train.py --model_type C --fold 1

- Train Model A (with augmentation) on fold 1
python train.py --model_type A --fold 1

Important options (with their defaults):
--model_type   A or C                # A = augmented, C = baseline
--fold         1..10                 # which fold is used as test fold
--batch_size   64
--lr           1e-3
--epochs       30
--weight_decay 1e-4
--save_dir     ../checkpoints

Each run will print training/validation loss and accuracy for each epoch, for example:
[C] Fold 1 Epoch 001 | train_loss=3.66 acc=0.17 | val_loss=1.97 acc=0.16
...
[C] Fold 1 TEST | loss=1.04 acc=0.63
The best validation checkpoint for this fold is saved as:
../checkpoints/best_C_fold1.pt
or ../checkpoints/best_A_fold1.pt
depending on --model_type.

- 3.3 training all folds
You can either write a small shell loop or use your own run_all_folds.py.

Example (bash loop, from /src):
- Train Model C on all 10 folds
for f in {1..10}; do
  python train.py --model_type C --fold $f
done

- Train Model A on all 10 folds
for f in {1..10}; do
  python train.py --model_type A --fold $f
done
After this, you should have:
../checkpoints/
    best_A_fold1.pt ... best_A_fold10.pt
    best_C_fold1.pt ... best_C_fold10.pt
# 4.Evaluation and visualization
Once all 10 folds have been trained for A and C, you can run the evaluation and visualization script:
cd /src
python eval_ac_all_folds.py --batch_size 64

This script will:
For each model_type in {C, A} and each fold 1..10:
Rebuild the test set using the official fold split.
Load the corresponding checkpoint best_{model_type}_fold{fold}.pt.
Compute test loss and accuracy.
Collect all predictions and labels across folds.
Aggregate results and create plots/CSV files under ../checkpoints/:
AC_test_acc_per_fold.png
Bar chart of test accuracy per fold (A vs. C).
AC_mean_test_acc.png
Mean test accuracy over 10 folds (A vs. C).
A_per_class_metrics.csv, C_per_class_metrics.csv
Per-class precision, recall, F1, and support.
A_confusion_matrix.png, C_confusion_matrix.png
10×10 confusion matrices for each model.
AC_per_class_f1.png
Side-by-side per-class F1 scores for A vs. C.
These files are exactly what we used to generate the figures and tables in the report.


#  Model B and D and E
1. Overview
These notebooks implement the remaining three models in our study:
Model B – 2D VGG13 on log-mel spectrograms, ImageNet pre-trained, no augmentation.
Model D – 1D VGG-style CNN on raw waveforms, scratch training with time-domain augmentation.
Model E – 2D VGG13 on log-mel spectrograms, ImageNet pre-trained with both time-domain and SpecAugment augmentation.
All three models assume the same UrbanSound8K dataset layout as in the A/C experiments.
2. Notebook files
The implementation is organized into the following Jupyter notebooks:
- Training (local)
train-model-b.ipynb – train Model B.
train-model-d.ipynb – train Model D.
train-model-e.ipynb – train Model E.
- Training (Kaggle versions)
train-model-b-kaggle.ipynb
train-model-d-kaggle.ipynb
train-model-e-kaggle.ipynb
These notebooks are adapted to Kaggle’s file paths and GPU environment.
- Evaluation & visualization
test-b.ipynb – evaluate Model B on test folds.
test-e.ipynb – evaluate Model E on test folds.
visualization-b-e.ipynb – plot accuracy/loss curves and other metrics for Models B and E.
3. Running locally
- Start Jupyter
conda activate urbansound
cd code/src
jupyter lab  # or jupyter notebook
- Open a notebook
In the browser, open one of:
train-model-b.ipynb

train-model-d.ipynb

train-model-e.ipynb
- Check imports and paths
In the first configuration cell, the notebook usually:
Adds the project root to sys.path so it can import dataset.py, augment.py, specAugment.py, and models.py.
Sets csv_path and audio_base_path (these should match the paths used in the A/C README).
If your dataset is in a different location, modify those variables accordingly.

- Run all cells
The notebook will:
Construct the UrbanSound8K dataset and dataloaders (often for a specific fold).
Build the appropriate model (B, D, or E).
Define loss and optimizer.
Train for the specified number of epochs and save checkpoints (usually into a checkpoints/ folder or a folder specified in the config cell).

4. Model-specific details
- Model B (pre-trained 2D VGG13, no augmentation)
Input: log-mel spectrograms, same preprocessing as Models A/C.
Backbone: torchvision.models.vgg13(pretrained=True) with the final classifier replaced by a 10-way linear layer.
No waveform or spectrogram augmentation; only deterministic preprocessing.
Purpose: isolate the effect of ImageNet pre-training compared to scratch Model C.
- Model D (1D VGG-style CNN on raw waveforms, with augmentation)
Input: raw waveforms padded/cropped to 4 seconds.
Backbone: custom 1D VGG-style network operating along the time axis.
Augmentation: uses the same WaveformAugment module as Model A (time shift, gain, noise, pitch, stretch).
Training is more expensive; some runs may use fewer epochs or partial freezing of early layers because of resource constraints.
Purpose: compare 1D vs. 2D architectures under similar augmentation.
- Model E (pre-trained 2D VGG13 + full augmentation)
Input: log-mel spectrograms.
Backbone: pre-trained VGG13 as in Model B.
Augmentation: both waveform-level (WaveformAugment) and spectrogram-level (SpecAugment) during training.
Purpose: test the combined effect of pre-training and strong augmentation, and compare against both B (pre-trained, no aug) and A (scratch, strong aug).

5. Evaluation and visualization
- Evaluating Model B
Open test-b.ipynb.
Set the checkpoint path(s) to match the files saved by train-model-b.ipynb.
Run all cells to compute per-fold test accuracy and any additional metrics.
Export summary statistics or plots as needed.
Per-fold test accuracy comparison between Models B and E.
Additional figures such as per-class F1 or confusion matrices (depending on implementation).
- Evaluating Model E
Open test-e.ipynb.
Adjust checkpoint paths to match train-model-e.ipynb.
Run all cells; the notebook will evaluate test accuracy on each fold and may compute per-class metrics.
Per-fold test accuracy comparison between Models B and E.
Additional figures such as per-class F1 or confusion matrices (depending on implementation).
- Visualizing results
Open visualization-b-e.ipynb.
Run all cells to produce:
Save the figures which are feature maps(e.g., as .png ) for use in reports or slides.

6. Running on Kaggle
- Upload the project and UrbanSound8K dataset to Kaggle (or add them as datasets to a Kaggle notebook).
- Choose one of the Kaggle notebooks:
train-model-b-kaggle.ipynb
train-model-d-kaggle.ipynb
train-model-e-kaggle.ipynb
- In the first cell, update csv_path and audio_base_path to Kaggle’s dataset paths (for example, /kaggle/input/urbansound8k/UrbanSound8K/metadata/...).
- Set the runtime to GPU and run all cells.
- Download the resulting checkpoints/metric files if you want to analyze them locally or combine them with the A/C evaluation scripts.