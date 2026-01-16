# **Fraudulent Job Prediction**
Online recruitment fraud has become increasingly prevalent in Malaysia due to rising living costs and the growing number of online job platforms. Many job seekers struggle to differentiate between legitimate and fraudulent job postings, leading to financial losses and loss of trust in digital hiring platforms.

This project aims to build a machine learning–based classification system to accurately detect fraudulent job postings using structured metadata and unstructured text features, while balancing predictive performance, computational efficiency, and real-world deployability.

**🛠️ Tools, Techniques & Platforms Used**

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **NLP Techniques:** TF-IDF Vectorization, Text Feature Engineering
- **Models:** Random Forest, Decision Tree, K-Nearest Neighbour (KNN), Linear SVC, SGD Classifier
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
- Filled remaining missing categorical values with “unspecified”
- Performed statistical analysis on text-based word counts
- Addressed class imbalance using random over-sampling
- Converted text features using TF-IDF vectorization
- Applied one-hot encoding to categorical variables
- Ensured final dataset contained only numerical features for model training
- Statistical significance testing showed that company profile and job requirements word counts are strong predictors of fraudulent postings.

### 3. Data Exploration
- Class Distribution Analysis
    - Real job postings were significantly underrepresented, confirming the need for resampling techniques.
    - <img src="image/Count of Fraudulent vs Real Job Posts.png" alt="Alt text" width="400" height="300"> 
- Feature Distribution Analysis
    - Distribution plots were used to compare fraudulent vs real job postings across features.
    - <img src="image/Count of Fraudulent vs Real Job Posts (Binary Features).png" alt="Alt text" width="900" height="300">
    - <img src="image/Count of Fraudulent vs Real Job Posts (Categorical Features).png" alt="Alt text" width="1200" height="400">   
- Text Feature Analysis
    - Word count distributions and hypothesis testing highlighted meaningful linguistic differences between fraudulent and legitimate job posts.

These analyses guided feature selection and preprocessing decisions for downstream modeling.
  

### 4. Modelling
The dataset was split into training and testing sets (80:20). Five supervised classification models were trained and compared:

- Random Forest
- Decision Tree
- K-Nearest Neighbour (KNN)
- Linear Support Vector Classifier (SVC)
- Stochastic Gradient Descent (SGD) Classifier

**Evaluation**
- Tree-based models performed better than linear regression, indicating the presence of nonlinear relationships.

    | Model | Accuracy | Percision | Recall | F1-Score |
    |------|-------------|-------------|-------|-------------| 
    | Random Forest | 99.95 | 99.95 |99.95 | 99.95 |
    | Decision Tree  | 98.81 | 98.83 | 98.81 | 98.81 |
    | KNN | 96.71 | 92.79 | 96.72 | 92.42 |
    | Linear SVC | 92.43 | 94.00 | 92.43 | 94.00 |
    | SGD | 73.60 | 74.49 | 73.60 | 73.36 |

   <br><br>
    <img src="image/Plots LR.png" alt="Alt text" width="300" height="300"> 
   <br><br>

- Tree-based models significantly outperformed linear models, indicating strong non-linear relationships in the dataset.


## 📊 Findings
- Random Forest achieved the best overall performance across all metrics and demonstrated strong robustness to noisy and high-dimensional features.
- Decision Tree provided good interpretability with competitive performance but required careful overfitting control.
- KNN showed high accuracy but suffered from scalability and computational cost issues.
- Linear models (Linear SVC, SGD) were efficient but underperformed on complex feature interactions.
- Text-based features, when combined with structured metadata, significantly improved fraud detection capability.

This project demonstrates that machine learning, combined with NLP techniques, can effectively detect fraudulent job postings with high accuracy. The deployed system provides a practical, low-cost solution suitable for real-world use.

Future enhancements include:
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