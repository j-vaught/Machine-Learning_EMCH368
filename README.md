# Machine Learning Exercises -- EMCH 368

A collection of four MATLAB machine learning exercises covering regression, text classification, digit recognition, and image classification. Each exercise includes datasets, training scripts, and pre-trained models so results can be reproduced without retraining.

## Requirements

**MATLAB R2023a or later** with the following toolboxes installed:

| Toolbox | Used In |
|---------|---------|
| Statistics and Machine Learning Toolbox | Exercises 1, 2 |
| Deep Learning Toolbox | Exercises 3, 4 |
| Image Processing Toolbox | Exercises 3, 4 |

Run `check_libraries.m` from the repository root to verify your installation.

## Exercises

### Exercise 1 -- Linear Regression (Housing Prices)

Implements gradient descent from scratch to fit a polynomial regression surface predicting home sale prices from square footage and year built.

```matlab
cd Exercise_1_regression
Machine_Learning_1a_home_data
```

**Dataset:** `train.csv` (Ames Housing data subset)

### Exercise 2 -- Text Sentiment Analysis (Random Forest)

Trains a 1000-tree Random Forest on pre-computed text embeddings to classify review sentiment. Includes OOB error visualization during training.

```matlab
cd Exercise_2_Text_Sentience
Text_sentience
```

Set `trainModel = true` on line 7 to retrain from scratch, or leave it `false` to load the included pre-trained model.

**Dataset:** `train_data.parquet`, `test_data.parquet`

### Exercise 3 -- Digit Classification (Neural Network)

Trains a three-layer pattern recognition network (300-200-100 hidden units) on the MNIST dataset.

```matlab
cd Exercise_3_Digit_Classification
MNIST_ML_Lab9
```

Set `trainModel = true` on line 12 to retrain, or leave it `false` to load the included model.

**Dataset:** `mnist.mat`

### Exercise 4 -- Image Classification (Residual Network)

Trains a ResNet on CIFAR-10 with data augmentation (random reflections, translations) for 80 epochs. Displays a confusion matrix and sample predictions on the validation set.

```matlab
cd Exercise_4_Image_Classification
CIFAR_10
```

If `trainedResidualNetwork.mat` is not present, the script will train automatically. Otherwise it loads the saved network.

**Dataset:** CIFAR-10 batches in `data/`

## Repository Structure

```
.
├── check_libraries.m              # Verify required toolboxes
├── Library_Check/
│   └── checkLibrary.m             # Alternate toolbox checker
├── Exercise_1_regression/
│   ├── Machine_Learning_1a_home_data.m
│   ├── preprocessAndSplitData.m
│   └── train.csv
├── Exercise_2_Text_Sentience/
│   ├── Text_sentience.m
│   ├── train_data.parquet
│   ├── test_data.parquet
│   └── trainedModel.mat
├── Exercise_3_Digit_Classification/
│   ├── MNIST_ML_Lab9.m
│   ├── mnist.mat
│   └── trainedModel.mat
└── Exercise_4_Image_Classification/
    ├── CIFAR_10.m
    ├── loadCIFARData.m
    ├── trainedResidualNetwork.mat
    └── data/
        ├── batches.meta.mat
        ├── data_batch_[1-5].mat
        └── test_batch.mat
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
