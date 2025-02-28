# Machine Learning Assignment 1 - Survival Prediction Post-HCT

<p align="center">
  <strong>The University of Azad Jammu and Kashmir, Muzaffarabad</strong><br>
  <em>Department of Software Engineering</em><br>
  <em>Bachelor of Science in Software Engineering (2022-2026)</em>
</p>

---

<table align="center" width="100%">
  <tr>
    <th>Course Details</th>
    <th>Submission Details</th>
  </tr>
  <tr>
    <td>
      <strong>Course Title:</strong> Machine Learning<br>
      <strong>Course Code:</strong> SE-3105<br>
      <strong>Instructor:</strong> Engr. Ahmed Khawaja<br>
      <strong>Semester:</strong> Fall 2024<br>
      <strong>Session:</strong> 2022 – 2026
    </td>
    <td>
      <strong>Submitted By:</strong><br>
      - Roll Number: 2022-SE-08<br>
      - Roll Number: 2022-SE-18<br>
      <strong>Degree Program:</strong> BS Software Engineering<br>
      <strong>Submitted To:</strong> Engr. Ahmed Khawaja<br>
      <strong>Date:</strong> February 28, 2025
    </td>
  </tr>
</table>

---

## 📖 Overview
This project predicts **Event-Free Survival (EFS)** for patients post-Hematopoietic Cell Transplantation (HCT) using two machine learning models: an **Artificial Neural Network (ANN)** and an **XGBoost model**. The dataset is sourced from the Kaggle competition "Equity Post-HCT Survival Predictions" and includes medical, genetic, and demographic features. The project demonstrates data preprocessing, model training, and prediction generation, culminating in submission files for evaluation.

### 📂 Project Files
- **Training (ANN)**: `training-hct-survival-ann.ipynb` - Trains an ANN model and saves it as `efs_model.pth`.
- **Testing (ANN)**: `testing-hct-survival-ann.ipynb` - Uses the pre-trained ANN to predict on the test set.
- **Training (XGBoost)**: `training-hct-survival-xgboost.ipynb` - Trains an XGBoost model and saves it as `xgboost_model.model`.
- **Testing (XGBoost)**: `testing-hct-survival-xgboost.ipynb` - Uses the pre-trained XGBoost model for predictions.

### 📊 Dataset
- **Source:** Kaggle competition (`train.csv`, `test.csv`)
- **Size:** 28,800 entries, 60 attributes
- **Target:** `efs` (binary: 0 = event, 1 = survival)
- **Key Features:** `prim_disease_hct`, `hla_match_b_low`, `prod_type`, `year_hct`, `obesity`, `donor_age`, `prior_tumor`, `gvhd_proph`, `sex_match`, `comorbidity_score`, `karnofsky_score`, `donor_related`, `age_at_hct`

<p align="center">
  <img src="media/image4.png" alt="Description of Data" >
  <br><em>Figure 1-a: Statistical Summary of Dataset (from Analysis Report)</em>
</p>
<p align="center">
  <img src="media/image5.png" alt="Description of Data" >
  <br><em>Figure 1-a: Statistical Summary of Dataset (from Analysis Report)</em>
</p>

---

## 🚀 Project Workflow
The project is structured into five key phases, executed across both ANN and XGBoost implementations:

1. **Data Loading**  
2. **Data Preprocessing**  
3. **Model Definition and Training**  
4. **Model Testing and Prediction**  
5. **Submission File Generation**

### 1. Data Loading
Loads training and test datasets from the Kaggle input directory.

#### ANN - Training
```python
import pandas as pd
train_file_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
df = pd.read_csv(train_file_path)
selected_columns = ["prim_disease_hct", "hla_match_b_low", "prod_type", "year_hct", "obesity", 
                    "donor_age", "prior_tumor", "gvhd_proph", "sex_match", "comorbidity_score", 
                    "karnofsky_score", "donor_related", "age_at_hct", "efs"]
df = df[selected_columns]
print(df.head())
```

#### XGBoost - Testing
```python
test_file_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
df_test = pd.read_csv(test_file_path)
selected_columns = ["prim_disease_hct", "hla_match_b_low", "prod_type", "year_hct", "obesity", 
                    "donor_age", "prior_tumor", "gvhd_proph", "sex_match", "comorbidity_score", 
                    "karnofsky_score", "donor_related", "age_at_hct"]
df_test = df_test[selected_columns]
```

---

### 2. Data Preprocessing
Handles missing values, encodes categorical variables, and scales numerical features.

<p align="center">
  <img src="media/image2.png" alt="Missing Values" >
  <br><em>Figure 2: Visualization of Missing Values (from Analysis Report)</em>
</p>

#### ANN - Training
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols.remove("efs")

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(df[cat_cols])
df_encoded = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
df = df.drop(columns=cat_cols).join(df_encoded)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

joblib.dump(num_imputer, "./preprocessor/num_imputer.pkl")
joblib.dump(cat_imputer, "./preprocessor/cat_imputer.pkl")
joblib.dump(encoder, "./preprocessor/encoder.pkl")
joblib.dump(scaler, "./preprocessor/scaler.pkl")
```

#### XGBoost - Testing
```python
num_cols = joblib.load("/kaggle/input/training-hct-survival/num_cols.pkl")
cat_cols = joblib.load("/kaggle/input/training-hct-survival/cat_cols.pkl")
num_imputer = joblib.load("/kaggle/input/training-hct-survival/num_imputer.pkl")
cat_imputer = joblib.load("/kaggle/input/training-hct-survival/cat_imputer.pkl")
encoder = joblib.load("/kaggle/input/training-hct-survival/encoder.pkl")
scaler = joblib.load("/kaggle/input/training-hct-survival/scaler.pkl")

for col in cat_cols:
    if col not in df_test.columns:
        df_test[col] = np.nan

df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])
encoded_cats_test = encoder.transform(df_test[cat_cols])
df_test_encoded = pd.DataFrame(encoded_cats_test, columns=encoder.get_feature_names_out(cat_cols))
df_test = df_test.drop(columns=cat_cols, errors='ignore').join(df_test_encoded)
df_test[num_cols] = scaler.transform(df_test[num_cols])
```

---

### 3. Model Definition and Training
Defines and trains the ANN and XGBoost models.

#### ANN - Training
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EFSModel(nn.Module):
    def __init__(self, input_size):
        super(EFSModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = EFSModel(input_size=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

for epoch in range(100):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}/100 - Train Loss: {train_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "/kaggle/working/efs_model.pth")
```

#### XGBoost - Training
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 6, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42
}

bst = xgb.train(params, dtrain, num_round=100, evals=[(dval, 'validation')])
bst.save_model("xgboost_model.model")
```

---

### 4. Model Testing and Prediction
Loads pre-trained models and generates predictions.

#### ANN - Testing
```python
X_test_tensor = torch.tensor(df_test.values, dtype=torch.float32).to(device)
model = EFSModel(input_size=df_test.shape[1]).to(device)
model.load_state_dict(torch.load("/kaggle/input/training-hct-survival/efs_model.pth", map_location=device))
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy().flatten()
```

#### XGBoost - Testing
```python
dtest = xgb.DMatrix(df_test)
bst = xgb.Booster()
bst.load_model("/kaggle/input/training-hct-survival/xgboost_model.model")
predictions = bst.predict(dtest)
```

---

### 5. Submission File Generation
Creates submission files with predictions.

#### ANN - Testing
```python
submission = pd.DataFrame({"ID": patient_ids, "prediction": predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'")
```

#### XGBoost - Testing
```python
submission = pd.DataFrame({"ID": df_test.index, "prediction": predictions})
submission.to_csv("submission.csv", index=False)
print("XGBoost predictions saved!")
```

---

## 📈 Results
- **ANN**: Trained for 100 epochs, validation loss reduced to 0.6129. Outputs probabilities via sigmoid.
- **XGBoost**: Trained for 100 rounds with log loss metric, leveraging gradient boosting for robust predictions.

<p align="center">
  <img src="media/image6.png" alt="Correlation Analysis" >
  <br><em>Figure 3: Correlation Analysis of Key Features (from Analysis Report)</em>
</p>

---

## 🔧 Improvements
- **Hyperparameter Tuning**: Adjust learning rates, ANN layer sizes, or XGBoost parameters (e.g., `max_depth`, `eta`).
- **Feature Engineering**: Introduce interaction terms or derived features.
- **Evaluation Metrics**: Add AUC-ROC or precision-recall curves.

<p align="center">
  <img src="media/image3.png" alt="Outliers" >
  <br><em>Figure 4: Outlier Detection (from Analysis Report)</em>
</p>

---

## 🛠️ How to Run
1. **Environment**: Kaggle Notebook, Python 3.10.12, libraries: `pandas`, `numpy`, `torch`, `xgboost`, `sklearn`, `joblib`.
2. **Steps**:
   - Run `training-hct-survival-ann.ipynb` to train the ANN.
   - Run `training-hct-survival-xgboost.ipynb` to train XGBoost.
   - Run `testing-hct-survival-ann.ipynb` for ANN predictions.
   - Run `testing-hct-survival-xgboost.ipynb` for XGBoost predictions.

---

<p align="center">
  <strong>Thank You!</strong><br>
  For inquiries, contact the Department of Software Engineering, UAJK.
</p>

---
