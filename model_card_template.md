# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a binary classification model built with scikit-learn RandomForestClassifier.  The pipeline one-hot encodes categorical features and binarizes the `salary` label to predict whether income is `<=50K` or `>50K`.

## Intended Use
This model is intended for educational use in demonstrating an end-to-end machine learning pipeline with training, evaluation, slice analysis, and API serving.  It is not inteded for real-world, high-stakes decisions such as hiring, lending, insurance, or legal adjudication.

## Training Data
The model trains on `data/census.csv`, which contains demographic and work-related features plus the `salary` target column.  Categorical features include `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-county`.
The original data has 32,563 rows and a 80-20 split was used for the train and test split.

## Evaluation Data
Evaluation is performed on a held-out test split from the same dataset after applying the training-fitted encoder with a label binarizer. 

## Metrics
The evaluation metrics are Precision, Recall, and F1 score on the held-out test split.
* Precison: `0.7353`
* Recall: `0.6378`
* F1: `0.6831`

## Ethical Considerations
This dataset includes sensitive and socio-economic attributes, so model performance may vary across groups and may reflect historical bias in the underlying data.

## Caveats and Recommendations
This model should be treated as a baseline.  Before any broader use, performance should be re-validated, slice disparities should be reviewed, and model/feature updates should be tested with stronger validation and monitoring.
