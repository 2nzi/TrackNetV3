# TrackNetV3 Football Adaptation

Ce projet est une adaptation de [TrackNetV3](https://github.com/qaz812345/TrackNetV3) pour le suivi de ballon de football. Il s'agit d'une modification du projet original qui était conçu pour le suivi de volants de badminton.

## Crédit et Référence

**Ce projet n'est PAS le mien**. Il s'agit d'une adaptation du travail original :

- **Projet Original** : [TrackNetV3](https://github.com/qaz812345/TrackNetV3)
- **Paper** : [TrackNetV3: A End-to-End Framework for Online Badminton Trajectory Prediction](https://dl.acm.org/doi/10.1145/3595916.3626370)
- **Auteurs Originaux** : Yi-Che Chen, Wei-Yih Chen, Wen-Chih Peng, et al.

## Installation

1. **Environnement requis**
```bash
Python 3.10.11
PyTorch (avec CUDA recommandé)
OpenCV
```

2. **Cloner le repository**
```bash
git clone [vot]
cd [votre-repo]
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Utilisation

1. **Télécharger les checkpoints**
```bash
# Télécharger et décompresser les checkpoints dans le dossier ckpts/
unzip TrackNetV3_ckpts.zip
```

2. **Prédiction sur une vidéo**
```bash
# Prédiction simple
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction

# Avec visualisation vidéo
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video

# Pour les vidéos volumineuses
python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --large_video --max_sample_num 20
```

## Modifications Apportées

Les principales modifications par rapport au projet original incluent :
- Adaptation pour le suivi de ballon de football
- Ajout d'affichage des coordonnées et de la confiance sur la vidéo
- Visualisation améliorée de la trajectoire
- Support pour les vidéos volumineuses
- Compatibilité avec Python 3.10.11

## Licence

Ce projet suit la même licence que le projet original TrackNetV3.

## Contact

Pour toute question concernant cette adaptation, veuillez ouvrir une issue dans ce repository.
Pour les questions concernant TrackNetV3, veuillez vous référer au [repository original](https://github.com/qaz812345/TrackNetV3).

<div align="center">
    <a href="./">
        <img src="./figure/NetArch.png" width="50%"/>
    </a>
</div>

## Performance 

* Performance on the test split of [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw).

<div align="center">
    <table>
    <thead>
        <tr>
        <th>Model</th> <th>Accuracy</th> <th>Precision</th> <th>Recall</th> <th>F1</th> <th>FPS</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>YOLOv7</td> <td>57.82%</td> <td>78.53%</td> <td>59.96%</td> <td>68.00%</td> <td><b>34.77</b></td>
        </tr>
        <tr>
        <td>TrackNetV2</td> <td>94.98%</td> <td><b>99.64%</b></td> <td>94.56%</td> <td>97.03%</td> <td>27.70</td>
        </tr>
        <tr>
        <td>TrackNetV3</td> <td><b>97.51%</b></td> <td>97.79%</td> <td><b>99.33%</b></td> <td><b>98.56%</b></td> <td>25.11</td>
        </tr>
    </tbody>
    </table>
    </br>
    <a href="./">
        <img src="./figure/Comparison.png" width="80%"/>
    </a>
</div>

## Training
### 1. Prepare Dataset
* Download [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
* Adjust file structure:
    1. Merge the `Professional` and `Amateur` match directories into a single `train` directory.
    2. Rename the `Amateur` match directories to start from `match24` through `match26`.
    3. Rename the `Test` directory to `test`.
* Dataset file structure:
```
  data
    ├─ train
    |   ├── match1/
    |   │   ├── csv/
    |   │   │   ├── 1_01_00_ball.csv
    |   │   │   ├── 1_02_00_ball.csv
    |   │   │   ├── …
    |   │   │   └── *_**_**_ball.csv
    |   │   ├── frame/
    |   │   │   ├── 1_01_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── 1_02_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── …
    |   │   │   └── *_**_**/
    |   │   │
    |   │   └── video/
    |   │       ├── 1_01_00.mp4
    |   │       ├── 1_02_00.mp4
    |   │       ├── …
    |   │       └── *_**_**.mp4
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    ├─ val
    |   ├── match1/
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    └─ test
        ├── match1/
        ├── match2/
        └── match3/
```
* Attributes in each csv files: `Frame, Visibility, X, Y`
* Data preprocessing
    ```
    python preprocess.py
    ```
* The `frame` directories and the `val` directory will be generated after preprocessing.
* Check the estimated background images in `<data_dir>/median`
    * If available, the dataset will use the median image of the match; otherwise, it will use the median image of the rally.
    * For example, you can exclude `train/match16/median.npz` due to camera angle discrepancies; therefore, the dataset will resort to the median image of the rally within match 16.
* Set the data root directory to `data_dir` in `dataset.py`.
    * `dataset.py` will generate the image mapping for each sample and cache the result in `.npy` files.
    * If you modify any related functions in `dataset.py`, please ensure you delete these cached files.
### 2. Train Tracking Module
* Train the tracking module from scratch
    ```
    python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training --verbose
    ```

### 3. Generate Predicted Trajectories and Inpainting Masks
* Generate predicted trajectories and inpainting masks for training rectification module
    * Noted that the coordinate range corresponds to the input spatial dimensions, not the size of the original image.
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16
    ```

### 4. Train Rectification Module
* Train the rectification module from scratch.
    ```
    python train.py --model_name InpaintNet --seq_len 16 --epoch 300 --batch_size 32 --lr_scheduler StepLR --mask_ratio 0.3 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name InpaintNet --epochs 30 --save_dir exp --resume_training
    ```

## Evaluation
* Evaluate TrackNetV3 on test set
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --split_list test
    python test.py --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval
    ```

* Evaluate the tracking module on test set
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval
    ```

* Generate video with ground truth label and predicted result
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --video_file data/test/match1/video/1_05_02.mp4 
    ```

## Error Analysis Interface
* Evaluate TrackNetV3 on test set and save the detail results for error analysis
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval --output_pred
    ```

* Add json path of evaluation results to the file list in `error_analysis.py`
    ```
    30  # Evaluation result file list
    31  if split == 'train':
    32      eval_file_list = [
    33          {'label': label_name, 'value': json_path},
     ⋮                              ⋮
            ]
        elif split == 'val':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        elif split == 'test':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        else:
            raise ValueError(f'Invalid split: {split}')                                  
    ```

* Run Dash application
    ```
    python error_analysis.py --split test --host 127.0.0.1
    ```
<div align="center">
    <a href="./">
        <img src="./figure/ErrorAnalysisUI.png" width="70%"/>
    </a>
</div>

## Reference
* TrackNetV2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
* Shuttlecock Trajectory Dataset: https://hackmd.io/@TUIK/rJkRW54cU
* Labeling Tool: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2?tab=readme-ov-file#label
