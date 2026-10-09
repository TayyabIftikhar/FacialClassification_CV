# Facial Classification Using Deep Learning

This project focuses on facial classification using deep learning models on a large-scale face recognition dataset. The work compares two prominent architectures: InceptionResNetV1 and EfficientNet-B0.

## Dataset

The dataset used in this project is taken from the Kaggle competition page:

https://www.kaggle.com/competitions/face-classification-deep-learning-cs-405/

This dataset is a large-scale facial recognition dataset designed for training and evaluating deep learning models for face classification.

### Dataset Summary
- Training set: more than 140,000 images
- Number of identities: 7,000 unique faces
- Images per identity: 20
- Test set: 35,000 images
- Test set structure: 5 images per identity, with unlabeled faces for prediction
- Usage: training, validation, and final prediction on unseen test data

The images are organized into folders by identity, which makes it suitable for supervised learning in a multi-class classification setup.

## Models Used

### 1) InceptionResNetV1

The InceptionResNetV1 model from the FaceNet PyTorch library was used.

- Pretrained on large face datasets
- Well-suited for facial recognition and classification tasks
- Uses inception-style convolution blocks with residual connections
- Fine-tuned for the specific facial identity classification task

#### Freeze Strategy
- Around 90% of the pretrained layers were frozen to retain the useful facial feature extractors learned from the original dataset.
- The last layers were unfrozen or adapted for the new face-classification task so the model could learn dataset-specific patterns.
- A custom classification head was added on top of the model to match the number of target identities.

#### Training Setup
- Train/validation split: 90/10
- Validation selection: 2 images from each class kept for validation
- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-5
- Loss function: CrossEntropyLoss
- Data augmentation: Albumentations-based augmentations such as horizontal flipping, rotation, and brightness/contrast adjustment

#### Results
- Training Accuracy: 91.61%
- Validation Accuracy: 89.69%
- Training Loss: 0.91
- Validation Loss: 0.94

This model showed strong and stable performance with a relatively small gap between training and validation accuracy, indicating good generalization.

### 2) EfficientNet-B0

The EfficientNet-B0 model was used with pretrained weights and a modified final classifier layer to match the number of target classes.

- Efficient and lightweight architecture
- Strong balance between accuracy and computational efficiency
- Fine-tuned for facial classification

#### Freeze Strategy
- The pretrained backbone was used as a feature extractor.
- Only the final classification layers were adapted for the new multi-class face recognition task.
- A custom fully connected output layer was added to classify the 7,000 identities.

#### Training Setup
- Train/validation/test split: 80/10/10
- Epochs: 10
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-5
- Loss function: CrossEntropyLoss
- Data augmentation: resizing, flipping, rotation, brightness adjustment, and normalization using ImageNet statistics

#### Results
- Training Accuracy: 97.91%
- Validation Accuracy: 73.70%
- Training Loss: 0.1022
- Validation Loss: 1.6598

Although the training accuracy was very high, the validation accuracy was noticeably lower, which suggests some overfitting. The model learned the training set very well, but its generalization to unseen validation data was weaker than that of the InceptionResNetV1 model.

## Comparative Result Summary

| Model | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|------|------------------|--------------------|--------------|-----------------|
| InceptionResNetV1 | 91.61% | 89.69% | 0.91 | 0.94 |
| EfficientNet-B0 | 97.91% | 73.70% | 0.1022 | 1.6598 |

## Conclusion

Both models were effective for the facial classification task, but they behaved differently:

- InceptionResNetV1 performed more consistently and showed better validation performance, making it more reliable for generalization.
- EfficientNet-B0 achieved higher training accuracy but showed overfitting tendencies, as seen in the large gap between training and validation performance.

Overall, InceptionResNetV1 was the stronger model for this dataset in terms of validation accuracy and generalization capability.

## How to Run

1. Upload the dataset from the Kaggle competition page and place it in a folder named `dataset`.
2. Open the notebook, install the required libraries, and run all cells in sequence to train and evaluate the models.

## Source

Dataset source: Kaggle competition page
- https://www.kaggle.com/competitions/face-classification-deep-learning-cs-405/
