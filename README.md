# Heart Disease Prediction Model

A machine learning project that predicts whether a patient has heart disease from clinical data. Four classifiers are trained and compared, and the SVM is saved as a ready-to-use model file.

## Project Structure

| File | Description |
|------|-------------|
| `heart.ipynb` | Jupyter notebook with data preparation, model training, evaluation and model export |
| `heart.csv` | Dataset used for training and testing (target column: `target`) |
| `heart_model.pkl` | Saved final model (full pipeline: preprocessing + SVM) |
| `requirement.txt` | Python dependencies |
| `README.md` | Project documentation |

## Data Preparation

- Features `X` are all columns except `target`; `y` is the `target` column.
- The data is split 80/20 into train and test sets (`test_size=0.2`, `random_state=42`, stratified on the target), giving 61 test samples.
- Preprocessing is built with a scikit-learn `ColumnTransformer` that combines two pipelines:
  - a **numeric pipeline** for numerical columns
  - a **categorical pipeline** for categorical columns
- The combined preprocessor is placed in front of every classifier inside a `Pipeline`, so the same cleaning is applied during training, evaluation and prediction.

## Models Compared

- Logistic Regression (`max_iter=1000`)
- Support Vector Machine (SVM, `probability=True`)
- Random Forest
- XGBoost

All models use `random_state=42` where applicable.

## Results

### 5-fold Stratified Cross-Validation (training data)

| Model | ROC-AUC (mean ± std) | Recall (mean ± std) |
|-------|----------------------|---------------------|
| Random Forest | 0.902 ± 0.036 | 0.809 ± 0.053 |
| SVM | 0.900 ± 0.039 | 0.800 ± 0.098 |
| Logistic Regression | 0.889 ± 0.041 | 0.764 ± 0.097 |
| XGBoost | 0.886 ± 0.028 | 0.782 ± 0.053 |

Random Forest and SVM are essentially tied on ROC-AUC (the difference is far smaller than the standard deviation), and both are ahead of Logistic Regression and XGBoost.

### Final model: SVM (test set, 61 samples)

```
              precision    recall  f1-score   support

  No disease       0.81      0.91      0.86        33
     Disease       0.88      0.75      0.81        28

    accuracy                           0.84        61
   macro avg       0.84      0.83      0.83        61
weighted avg       0.84      0.84      0.83        61
```

Overall accuracy is 0.84 with a balanced trade-off between the two classes.

### Update: corrected class labels

An earlier version of the notebook had the class labels mixed up, which showed the wrong class distribution (28 / 33) and inflated the "Disease" recall to 0.97. This is now fixed: the test set has 33 "No disease" and 28 "Disease" samples, the metrics above reflect the corrected labels, and `heart_model.pkl` was re-saved ("Saved SVM with corrected labels"). If you downloaded the model before this fix, replace it with the new file.

## Installation

```bash
pip install -r requirement.txt
```

Main libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`. Developed with Python 3.13.

## Usage

### Run the notebook

Open `heart.ipynb` in Jupyter or VS Code and use **Run All**. The notebook will:

1. Load `heart.csv` and split it into train and test sets
2. Build the preprocessing pipeline (numeric + categorical)
3. Train all four models and compare their metrics (accuracy, precision, recall, F1, ROC-AUC) in a table and bar chart
4. Run 5-fold stratified cross-validation
5. Fit the final models, print the SVM classification report and save the SVM to `heart_model.pkl`

### Use the saved model

```python
import pickle
import pandas as pd

with open("heart_model.pkl", "rb") as file:
    model = pickle.load(file)

# One row per patient, with the same columns as the training features in heart.csv
new_patients = pd.DataFrame([{
    # "age": 54, "sex": 1, ...   <- fill in your feature columns here
}])

prediction = model.predict(new_patients)          # predicted class (No disease / Disease)
probability = model.predict_proba(new_patients)   # class probabilities

print(prediction, probability)
```

Because the pipeline includes preprocessing, pass raw feature values in the same format as the original data (do not scale or encode them yourself).

## Notes and Limitations

- The test set is small (61 samples), so results can vary; the cross-validation scores give a more stable estimate.
- Recall for the "Disease" class is 0.75, so roughly 1 in 4 patients with heart disease is missed on the test set. For a screening use case you may want to lower the decision threshold to raise recall.
- Only load `.pkl` files from sources you trust, since pickle can execute arbitrary code.
- Use the same scikit-learn version for loading the model as was used to save it.

## Disclaimer

This project is for learning and educational purposes only. It is **not** a medical device and must not be used for real diagnosis or treatment decisions. Always consult a qualified healthcare professional.
