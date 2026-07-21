# **Fraudulent Job Prediction**
 Online recruitment fraud is on the rise in Malaysia, fueled by growing living costs and a flood of new job platforms competing for attention. Scammers exploit this — posting fake listings that mimic real companies, promise inflated salaries, or ask for upfront "processing fees." For job seekers already under financial pressure, telling a legitimate posting apart from a fraudulent one isn't always easy, and the consequences of getting it wrong range from lost money to lasting distrust in online hiring platforms altogether.

## Objective:
 Build a machine learning classification system that can flag fraudulent job postings by combining structured metadata (salary range, location, and company profile) with unstructured text features from job descriptions. The model has to strike a practical balance between accuracy, computational cost, and real-world deployability.
 

## ⚙️ Project Structure /method
**🛠️ Tools, Techniques & Platforms Used**

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **NLP Techniques:** TF-IDF Vectorization, Text Feature Engineering
- **Models:** Random Forest, Decision Tree, XGB Classifier, SGD Classifier
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Depolyment Tools:** GitHub, Google Colab, [Streamlit](https://fraudulent-job-prediction-404.streamlit.app/)

### 1. Dataset
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and contains 17,880 job postings with 17 features and 1 binary target variable indicating whether a job posting is fraudulent.

The dataset consists of a mix of categorical, boolean, numerical, and text-based features, making it suitable for both traditional machine learning and NLP-based approaches.

The key attributes in the dataset are:
- **Job metadata:** title, location, department, salary range, employment type, benefit
- **Company information:** company profile, industry, function
- **Target variable:** fraudulent (0 = real, 1 = fraud)

### 2. Preprocessing
The following preprocessing steps were applied:
- Removed rows with excessive missing values (<10% of total dataset)
- Filled remaining missing categorical values with “none”
- Performed statistical analysis on text-based word counts
- Converted text features using TF-IDF vectorization
- Applied one-hot encoding to categorical variables
- Ensured final dataset contained only numerical features for model training
- Evaluate the model performance and addressed class imbalance using class weightning and threshold tunning , followed by hyperparameter tuning. 

### 3. Data Exploration
- Class Distribution Analysis
    - Real job postings were significantly underrepresented, confirming the need for resampling techniques.
    - <img src="image/Count of Fraudulent vs Real Job Posts.png" alt="Alt text" width="400" height="300"> 

- Feature Distribution Analysis
    - Distribution plots were used to compare fraudulent vs real job postings across binary, catergorical, text features.
    - <img src="image/Count of Fraudulent vs Real Job Posts (Binary Features).png" alt="Alt text" width="600" height="400">
    - <img src="image/Count of Fraudulent vs Real Job Posts (Categorical Features).png " alt="Alt text" width="700" height="600">   
    - <img src="image/Count of Fraudulent vs Real Job Posts (Categorical Features)_2.png " alt="Alt text" width="800" height="250">  
- Text Feature Analysis
    - Word count distributions and hypothesis testing highlighted meaningful linguistic differences between fraudulent and legitimate job posts.
    - <img src="image/Fraudulent Job Postings Word Cloud.png " alt="Alt text" width="500" height="250"> 
    - <img src="image/Non-Fraudulent Job Postings Word Cloud.png " alt="Alt text" width="500" height="250"> 

These analyses guided feature selection and preprocessing decisions for downstream modeling.
  

### 4. Modelling
The dataset was split into training and testing sets using an 80:20 ratio. Four supervised classification models were trained and evaluated:

- Random Forest
- Decision Tree
- eXtreme Gradient Boosting  (XGB) Classifier
- Stochastic Gradient Descent (SGD) Classifier

**Model Performance**

## 📊 Findings
### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | PR_AUC | Execution time (s) |
|-------|----------|-----------|--------|----------|--------|--------------------|
| Random Forest | 98.40 | 98.29 | 68.04 | 80.42 | 99.28 | 94.53 | 3.55 |
| XGB | 98.69 | 96.90 | 76.92 | 84.97| 99.47 | 93.45 | 92.45 |
| Decision Tree | 97.83 | 77.51 | 77.51 | 77.51 | 88.19 | 61.17 | 12.29 |
| SGD | 95.84 | 75.56 | 20.12 | 31.78 | 82.66 | 42.02 | 1.14 |

- **XGBoost** provides the highest F1-Score (84.97%) and PR_AUC, making it the most robust model for handling class imbalance. But its execution time is ~26x slower than Random Forest.
- **Random Forest** maintains elite precision (98.29%) and a strong PR_AUC while remaining computationally efficient.
- **Final Model Selection**: Random Forest was chosen for tuning based on comparable performance to XGBoost (similar F1-score and PR-AUC) with faster execution time.

### Handling Class Imbalance & Hyperparameter Tuning 
To address the class imbalance in fraudulent job postings, two strategies were applied:
- **Class Weighting**: Applied `class_weight='balanced_subsample'` to penalize misclassification of minority (fraud) cases
- **Threshold Tuning**: Tested multiple decision thresholds to optimize the precision-recall trade-off
- Tuned Random Forest model with parameters below
- `class_weight='class_weight='balanced_subsample'` , `decision threshold = 0.4 `,`'max_features': 'log2'`, `'min_samples_split': 2`, `'n_estimators': 300`


This project demonstrates that machine learning, combined with NLP techniques, can effectively detect fraudulent job postings. The deployed system provides a practical, low-cost solution suitable for real-world use.

### Future enhancements include:
- Hyperparameter optimization and feature selection refinement
- Incorporating deep learning models for text understanding
- Real-time monitoring and automated model retraining
- Expansion to multilingual job postings

## 🚀 Deployment
The best-performing model was deployed using a Streamlit web application, allowing users to input job posting details and fraud predictions.

🔗 [Live App](https://fraudulent-job-prediction-404.streamlit.app/)


## References
- Pablo, Guillermo & Alberto. (2023). Fake Job Detection with Machine Learning: A Comparison.

- Nasteski, V. (2017). An overview of supervised machine learning methods.

- Wang, X., Yan, L., & Zhang, Q. (2021). Application of gradient descent in ML.