# Product Category Classifier

## Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to predict the category of an e-commerce product based on its title.

The goal is to build a text classification model that can automatically assign product categories using product titles.

## Dataset

The project uses an e-commerce products dataset containing more than 30,000 product records.

The dataset includes information such as:
- Product ID
- Product Title
- Merchant ID
- Category Label
- Product Code
- Number of Views
- Merchant Rating
- Listing Date

The main features used for classification are product titles, while the target variable is the product category.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook / Google Colab
- TF-IDF
- Machine Learning

## Project Workflow

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Text feature extraction using TF-IDF
4. Splitting the dataset into training and testing sets
5. Training a Machine Learning classification model
6. Model evaluation using Accuracy, Precision, Recall and F1-score

## Machine Learning Model

The project uses Logistic Regression as the initial classification algorithm.

## Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Project Structure

```text
product-category-classifier/
├── README.md
├── product_category_analysis.ipynb
├── train_model.py
├── predict_category.py
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Install the required dependencies.
3. Open the Jupyter Notebook.
4. Upload the dataset.
5. Run the notebook cells.

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Status

The data preprocessing, feature extraction, model training and initial evaluation stages have been completed in Google Colab.

The remaining project files and final model export will be added as the project is finalized.
