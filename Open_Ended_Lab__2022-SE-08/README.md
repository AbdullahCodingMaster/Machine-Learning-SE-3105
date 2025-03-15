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
Four models were trained:
1. **Logistic Regression**: `max_iter=1000`.
2. **K-Nearest Neighbors (KNN)**: Default parameters with k=5.
3. **Naive Bayes**: Gaussian Naive Bayes.
4. **Artificial Neural Network (ANN)**: Simple MLP with one hidden layer (100 neurons).

##### Hyperparameter Tuning
ANN was tuned with:
- `hidden_layer_sizes`: [(100,), (50, 50)]
- `learning_rate_init`: [0.001, 0.01]
- Cross-validation: 3 folds on a 5,000-sample subset.

##### Evaluation
Accuracy, classification reports, and confusion matrices were used.

#### Results
| Model                       | Accuracy  |
|-----------------------------|-----------|
| Logistic Regression         | 0.140     |
| KNN                        | 0.839     |
| Naive Bayes                | 0.602     |
| ANN                        | 0.903     |
| Tuned ANN                  | 0.914     |

- **Logistic Regression**: 0.140, poor performance, likely due to insufficient capacity for complex patterns.
- **KNN**: 0.839, strong distance-based classification, benefiting from feature selection.
- **Naive Bayes**: 0.602, moderate performance, limited by feature independence assumptions.
- **ANN**: 0.903, high accuracy with a simple architecture; tuned version at 0.914 with `hidden_layer_sizes=(100,), learning_rate_init=0.01`.

**Visualizations**:
- Bar plot showed Tuned ANN leading.
- Confusion matrices indicated errors (e.g., 5 vs. 3).

#### Discussion
The Tuned ANN excelled with 0.914 accuracy, leveraging its ability to model complex, non-linear patterns through hyperparameter optimization. Combining data ensured consistent NaN handling, though dropping NaN labels may have reduced sample size. Feature selection to 200 dimensions maintained performance, with KNN adapting well due to its reliance on local structure. Logistic Regression struggled significantly (0.140), possibly due to the reduced feature set or linear assumptions not suiting the MNIST data. Naive Bayes (0.602) performed moderately but was hindered by its assumption of feature independence, which doesn’t hold for pixel data. Preprocessing was critical for all models, and future work could explore deeper ANN architectures or retaining more features to boost performance further.

#### Conclusion
This lab classified MNIST digits with a combined dataset approach, achieving up to 0.914 accuracy with Tuned ANN. The process highlighted the importance of NaN handling, feature selection, and model choice in an open-ended setting. While simpler models like Logistic Regression underperformed, ANN and KNN demonstrated robustness, offering a flexible framework for classification tasks.