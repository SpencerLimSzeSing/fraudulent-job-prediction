# **Fraudulent Job Prediction**
Online recruitment fraud has become increasingly prevalent in Malaysia due to rising living costs and the growing number of online job platforms. Many job seekers struggle to differentiate between legitimate and fraudulent job postings, leading to financial losses and loss of trust in digital hiring platforms.

This project aims to build a machine learning–based classification system to accurately detect fraudulent job postings using structured metadata and unstructured text features, while balancing predictive performance, computational efficiency, and real-world deployability.

**🛠️ Tools, Techniques & Platforms Used**

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **NLP Techniques:** TF-IDF Vectorization, Text Feature Engineering
- **Models:** Random Forest, Decision Tree, XGB Classifier, SGD Classifier
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Resampling Technique:** Random Over-Sampling
- **Depolyment Tools:** GitHub, Google Colab, Streamlit

## ⚙️ Project Structure /method

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
| XGB | 98.20 | 96.49 | 65.09 | 77.74 | 88.36 | 51.70 |
| Random Forest | 98.06 | 100.00 | 59.76 | 74.81 | 86.52 | 4.31 |
| Decision Tree | 96.89 | 66.67 | 71.00 | 68.77 | 48.73 | 11.04 |
| SGD | 86.97 | 21.20 | 62.72 | 21.69 | 16.83 | 0.47 |

- **XGBoost** delivered the best balance between recall and precision, achieving the highest F1-score and PR-AUC. But it required significantly longer execution time.
- **Random Forest** achieved perfect precision (1.00), making virtually no false fraud predictions, but had lower recall (59.76%), missing some fraudulent postings.
- **Final Model Selection**: Random Forest was chosen for tuning based on comparable performance to XGBoost (similar F1-score and PR-AUC) with faster execution time.

### Handling Class Imbalance & Hyperparameter Tuning 
To address the class imbalance in fraudulent job postings, two strategies were applied:
- **Class Weighting**: Applied `class_weight='balanced'` to penalize misclassification of minority (fraud) cases
- **Threshold Tuning**: Tested multiple decision thresholds to optimize the precision-recall trade-off
- Tuned Random Forest model with parameters below
- `class_weight='balanced'` , `decision threshold = 0.2 `,`'max_features': 'sqrt'`, `'min_samples_split': 2`, `'n_estimators': 200`
- Final tuned model performance: 

| Model | Accuracy | Precision | Recall | F1-Score | PR_AUC | Execution time (s) |
|-------|----------|-----------|--------|----------|--------|--------------------|
| Random Forest | 98.26 | 81.03 | 83.43 | 82.22 | 89.61 | 17.44 |

- Significantly improved recall (≈60% → 83%)
- Maintained strong precision (>80%)
- Achieved a better F1-score and PR-AUC
- Provided a more balanced fraud detection capability

This project demonstrates that machine learning, combined with NLP techniques, can effectively detect fraudulent job postings. The deployed system provides a practical, low-cost solution suitable for real-world use.

### Future enhancements include:
- Hyperparameter optimization and feature selection refinement
- Incorporating deep learning models for text understanding
- Real-time monitoring and automated model retraining
- Expansion to multilingual job postings

## 🚀 Deployment
The best-performing Random Forest model was deployed using a Streamlit web application, allowing users to input job posting details and receive real-time fraud predictions.

Deployment Platform: Streamlit Cloud
Repository Hosting: GitHub
Prediction Time: ~30 seconds
Reported Accuracy: 99.95%

🔗 [Live App](https://fakejobsdeploy-wqd7006-group15.streamlit.app)


## References
- Pablo, Guillermo & Alberto. (2023). Fake Job Detection with Machine Learning: A Comparison.

- Nasteski, V. (2017). An overview of supervised machine learning methods.

- Wang, X., Yan, L., & Zhang, Q. (2021). Application of gradient descent in ML.