# More complex encoder is not all you need

![image](./image/neUNet.png)

Unlike most methods that simply focus on building powerful encoder, we shift our attention to other "more meaningful" aspects and create neU-Net (i.e., not complex encoder U-Net). Our network seamlessly integrates with the nnU-Net[1] framework and has achieved state-of-the-art results on the Synapse multiorgan segmentation[2] and Automatic Cardiac Diagnosis Challenge (ACDC)[3] datasets.

---
## Environmental Configuration
#### 1. System requirements

We run neU-Net on Ubuntu 20.04.5 with Python 3.9, PyTorch 2.0.0, CUDA 11.6, and nnU-Net 2.1.1.All experiments are conducted on a single NVIDIA
GeForce RTX 3090 GPU with 24 GB memory.For specific hardware and software requirements, you can refer to:[Installation_Instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md).

#### 2. Installation guide
You can install neU-Net using the following steps:

- `git clone https://github.com/aitechlabcn/neUNet.git`
- `cd neUNet`
- ``pip install -e .``

#### 3. Dataset format

Our network operates within the nnU-Net 2.1.1 framework, so the dataset format aligns with nnU-Net as follows:

```
.../DATASET/
    ├── nnUNet_raw/
        ├── DatasetID_NAME/
            ├── imagesTr/
                ├── ...
            ├── imagesTs/
                ├── ...          
            ├── labelsTr/
                ├── ...
            ├── labelsTs/
                ├── ...          
            ├── dataset.json
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
```

Where nnUNet_raw, nnUNet_preprocessed, and nnUNet_results respectively store the raw dataset, preprocessed data, and training results.

For each dataset name in the format DatasetID_NAME: ID is a 3-digit identifier and NAME is the dataset name.Each raw dataset comprises these components:

- **imagesTr** contains training images.
- **labelsTr**  contains the corresponding labels for the training images.
- **imagesTs** is optional and contains test set cases.
- **labelsTs** is also optional and contains the labels for the test set(if available).
- **dataset.json** contains metadata of the dataset.

#### 4. Setting up paths

Similar to nnU-Net, it's necessary to set environment variables to specify the paths for raw files, preprocessed data, and result files.

For Linux systems, you need to add the following lines to your **.bashrc** file in your HOME directory:

```
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Make sure to replace "/path/to" with the actual paths to the nnUNet raw data, preprocessed data, and results directories. After adding these lines, save the **.bashrc** file, and the environment variables will be set the next time you open a terminal session.

#### 5. Dataset download

The download links for the datasets used in the experiments are as follows:

- **ACDC**    [Link for ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
- **Synapse**    [Link for Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).
- **BTCV**    [Link for BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).

---

## Experiment planning and preprocessing

You can obtain dataset information, training plans, and preprocessed data using the following commands:

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Where `DATASET_ID` is the dataset id. When preprocessing the dataset for the first time, you can use  `--verify_dataset_integrity` to check for dataset integrity and other common errors.

The **plans.json** file that is outputted includes commonly used hyperparameters for training, including network name (default: neUNet), network structure, batch size, patch size, and more. 

The plan.json files for the three datasets we use are stored in the **neUNet/neUNet_utils/plan_json/** directory.

## Training

You can train the network using the following command:

```
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD --wavelets [--npz]
```

Where `DATASET_NAME_OR_ID` is the id or name of dataset. When preprocessing the dataset for the first time, you can use  `--verify_dataset_integrity` to check for dataset integrity and other common errors. 

Since we are employing a five-fold cross-validation for training the network, you will need to select `FOLD`, which should be an integer between 0 and 4. The five-fold split is implemented by reading the **splits_final.json"**file. If there is no predefined five-fold split, the data will be randomly partitioned when training begins, and the **splits_final.json** file will be generated in the **nnUNet_preprocessed** directory to record the splits, splits_final.json files for the three datasets we use are stored in the **neUNet/neUNet_utils/split_json/** directory.

`--wavelets` represents the introduction of multi-scale wavelets at the input stage. 

`--npz` is optional and allows the model to save the softmax outputs while preserving the inference results during the final validation.

You can click the  [link ](https://drive.google.com/drive/folders/1NGZQAOoA9sy6XBRhOouF8ydMCSyx2s8E) to download our training weights.

## Prediction

You can perform inference using the following command:

```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -f FOLD --wavelets [--save_probabilities]
```

Where `INPUT_FOLDER`  and `OUTPUT_FOLDER` represent the input and output folder paths, respectively.`FOLD` can be an integer between 0 and 4 to specify a particular fold or `all ` to use all 5 folds for the operation.

`----save_probabilities` is optional and allows the model to save the softmax outputs while preserving the prediction results during inference.

## Analyze

We evaluate the model based on MONAI[4] using the Dice Similarity Coefficient (DSC) and 95th percentile Hausdorff Distance (HD95), just run:

```
Analyze --image_path IMAGE_FOLDER --predict_path PREDICT_FOLDER --label_path LABEL_FOLDER --dataset_json CONFIG_FOLDER 
--save_path OUTPUT_FOLDER --num_classes CLASSES --dataset_name NAME
```

Where  `CLASSES`  represents the number of classes in the dataset, and   `NAME ` represents the dataset's name.

## References

[1]: Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211.

[2]: Landman, Bennett, et al. "Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge." Proc. MICCAI Multi-Atlas Labeling Beyond Cranial Vault—Workshop Challenge. Vol. 5. 2015.

[3]: O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and
Diagnosis: Is the Problem Solved ?" in IEEE Transactions on Medical Imaging.

[4]: Cardoso, M. Jorge, et al. "Monai: An open-source framework for deep learning in healthcare." *arXiv preprint arXiv:2211.02701* (2022).

