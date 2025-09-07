# ViT_CatDog_classification
> This project demonstrates how to train a Vision Transformer (ViT) from scratch for a binary image classification task: distinguishing between cats and dogs.

## 📂 Project Structure
```
├── notebook.ipynb           # Training + evaluation notebook
├── data/                    # Dataset (downloaded from Kaggle)
└── README.md                # Project documentation
```

## Dataset Preparation
We use the Dog and Cat Classification Dataset from Kaggle:  
[Kaggle Dataset: Dog and Cat Classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)  

Steps:
1. Create a Kaggle account
2. Choose link above, download manually

## Model: Vision Transformer (ViT)
Unlike CNNs that extract features hierarchically, Vision Transformers split images into patches and process them like sequences of tokens, similar to NLP Transformers.

> [!TIP]
> Key Components in Our Implementation  

**Patch + Positional Embedding**  
Images are divided into patches (e.g., 16x16). Each patch is projected into a vector (embed_dim) and combined with a learnable positional embedding to retain spatial information.

```
self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                       kernel_size=patch_size, stride=patch_size, bias=False)
self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, embed_dim))
```

**Transformer Encoder**  
Multiple self-attention + feed-forward layers to learn dependencies between patches  

**Classification Head**  
The [CLS] token representation is passed through fully connected layers to output 2 classes (cat/dog).  

## Training & Evaluation
We train and evaluate everything inside the provided Jupyter Notebook ([ViT.ipynb](ViT.ipynb))

**Training Workflow:**  
1. Load dataset and preprocess images (resize, normalization).  
2. Initialize the Vision Transformer model.  
3. Train with CrossEntropyLoss and Adam optimizer.  
4. Validate accuracy on the test set.  

> At the end, you will obtain training/validation accuracy and loss plots.

## Acknowledgements
[Kaggle Dataset: Dog and Cat Classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)  
Vision Transformer concept from “An Image is Worth 16x16 Words” (Dosovitskiy et al., 2020)