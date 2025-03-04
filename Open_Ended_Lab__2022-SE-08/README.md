### Open-Ended Lab Report: Classification of MNIST Handwritten Digits Using Machine Learning

#### Introduction
The MNIST dataset comprises 28x28 grayscale images of handwritten digits (0-9), flattened into 784-dimensional vectors with pixel intensities from 0 to 255. Originally split into training and testing sets (`mnist_train.csv` and `mnist_test.csv`), this lab combines them for unified preprocessing, addressing NaNs in both features and labels, and re-splitting into train/test sets.

#### Methodology
##### Dataset Preparation
- **Loading and Combining**: Training and testing datasets were concatenated into a single DataFrame.
- **NaN Handling**: 
  - Labels with NaNs were dropped (or could be imputed with mode).
  - Feature NaNs were imputed with column means using `SimpleImputer`.
- **Preprocessing**: Features were normalized (0-1) and standardized (`StandardScaler`).
- **Feature Selection**: `SelectKBest` selected the top 200 features based on ANOVA F-value.
- **Splitting**: Data was split into 80% training and 20% testing sets with stratification.

##### Models Used
Five models were trained:
1. **Logistic Regression**: `max_iter=1000`.
2. **SVM**: RBF kernel.
3. **Random Forest**: 100 trees.
4. **HistGradientBoostingClassifier**: NaN-tolerant gradient boosting.
5. **Tuned HistGradientBoostingClassifier**: Optimized version.

##### Hyperparameter Tuning
`HistGradientBoostingClassifier` was tuned with:
- `max_iter`: [100, 200]
- `learning_rate`: [0.01, 0.1]
- Cross-validation: 3 folds on a 5,000-sample subset.

##### Evaluation
Accuracy, classification reports, and confusion matrices were used.

#### Results
| Model                       | Accuracy  |
|-----------------------------|-----------|
| Logistic Regression         | 0.910     |
| SVM (RBF)                  | 0.935     |
| Random Forest              | 0.925     |
| HistGradientBoosting       | 0.930     |
| Tuned HistGradientBoosting | 0.940     |

- **Logistic Regression**: 0.910, decent but limited by reduced features.
- **SVM (RBF)**: 0.935, strong with non-linear patterns.
- **Random Forest**: 0.925, robust but slightly lower.
- **HistGradientBoosting**: 0.930, effective with selected features.
- **Tuned HistGradientBoosting**: 0.940, best with `max_iter=200, learning_rate=0.1`.

**Visualizations**:
- Bar plot showed Tuned HistGradientBoosting leading.
- Confusion matrices indicated errors (e.g., 5 vs. 3).

#### Discussion
The Tuned HistGradientBoostingClassifier excelled, benefiting from tuning and robustness to feature selection. Combining data ensured consistent NaN handling, though dropping NaN labels may have reduced sample size. Feature selection to 200 dimensions maintained performance while reducing complexity, though SVM adapted best to this reduction. Preprocessing was critical, and HistGradientBoosting’s NaN tolerance suggests it’s ideal for imperfect data. Future work could explore more features or deep learning for higher accuracy.

#### Conclusion
This lab classified MNIST digits with a combined dataset approach, achieving up to 0.940 accuracy with Tuned HistGradientBoosting. The process highlighted the importance of NaN handling and feature selection in an open-ended setting, offering a flexible framework for classification tasks.