# **Crop Yield Prediction**
Unpredictable climate conditions have significantly affected agricultural productivity, contributing to global food insecurity (World Food Program USA, 2024). This project aims to identify key drivers of global crop yield and to forecast yields across different locations and crops. The workflow includes descriptive analysis, correlation analysis, feature engineering, and machine learning modeling. 

**🛠️ Tools, Techniques & Platforms Used**

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models:** Random Forest, Linear Regression, Decision Tree
- **Evaluation Metrics:** R² score ,MAE, MSE, RMSE
- **Resampling Technique:** Bootstrap

## ⚙️ Project Structure /method

### 1. Dataset
The datasets are sourced from the [World Bank](https://data.worldbank.org/) available on [Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data). The dataset is provided under the **World Bank Dataset Terms of Use**.  Please refer to the [World Bank Terms](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets) for details on permitted usage.

This dataset is a compiled agricultural dataset containing historical crop yield records with environmental variables across multiple countries and years.

The attributes in the dataset are:
- **Area**: Country or region where the crop is produced
- **Item**: Type of crop (e.g., maize, wheat, rice)
- **Year**: Year of observation
- **Yield**: Crop yield measured in hectograms per hectare (hg/ha)
- **Rainfall**: Average annual rainfall (mm)
- **Pesticides**: Amount of pesticide usage (tonnes)
- **Temp**: Average annual temperature (°C)

### 2. Preprocessing
The following preprocessing steps were applied:
- Removed unnecessary metadata columns (Area Code, Domain, Element Code, etc.)
- Renamed column for better readability and understanding.
- Handled missing values by removing incomplete rows
- Data formatting
- Combined yield, rainfall, temperature, and pesticide dataframe by year and country
- Data trimming by identify the value of the 95th percentile and remove outliers
- Encoded categorical variables (country, crop type) using one-hot encoding

### 3. Data Exploration
- Data Distribution
    - Histograms and boxplots were used to analyze the distributions of crop yield, rainfall, temperature, and pesticide usage.
        - Yield & Pesticide is right skewed
        - <img src="image/data distribution (histogram).png" alt="Alt text" width="500" height="500">   
- Correlation Analysis
    - A correlation heatmap was used to identify relationships between numerical variables.
        - There are correlation between Area and pesticides_tonnes, followed by Area and average rainfall
        - There are correlation between Item and hg/ha_yield
        - <img src="image/correlation heatmap.png" alt="Alt text" width="400" height="400">   
- Trend Analysis
    - Time series plots were created to analyze crop yield trends across years.
        - The yield and pesticides used increases graduallly over time
        - <img src="image/trend analysis.png" alt="Alt text" width="500" height="300">   
- Descriptive Analysis
    - Top 10 Countries by Total Yield (log scale)
        - <img src="image/Top 10 Countries by Total Yield (log scale).png" alt="Alt text" width="500" height="300">    
    - Top 20 Total Yields by Item
        - <img src="image/Top 10 Total Yields by Item.png" alt="Alt text" width="500" height="300">    

### 4. Modelling
The dataset was split into training and testing sets (80:20). The following models were evaluated:
- Linear Regression
- Decision Tree
- Random Forest

**Evaluation**
- Tree-based models performed better than linear regression, indicating the presence of nonlinear relationships.

    | Model | Description | Performance | R2_score | MSE | RMSE | MAE |
    |------|-------------|-------------|------|-------------|-------------|-------------| 
    | Linear Regression | Baseline linear model | Moderate | 0.807 | 0.224 | 0.473 | 0.356 |
    | Decision Tree | Captures non-linear patterns | High | 0.966 | 0.039 |	0.198 |	0.068 |
    | Random Forest | Ensemble tree-based model | Very High | 0.981 |	0.022	| 0.149	| 0.063 |
   
   <br><br>
    <img src="image/Plots LR.png" alt="Alt text" width="300" height="300"> 
    <img src="image/Plots DT.png" alt="Alt text" width="300" height="300"> 
    <img src="image/Plots RF.png" alt="Alt text" width="300" height="300">    
    <br><br>

- Feature Importance Stability
    - Feature importance was evaluated using Random Forest with repeated subsampling
    - Mean importance and MAD (Mean Absolute Deviation) were computed to assess both relevance and stability.


## 📊 Findings
- Model with the best performance is **Random Forest** after further hypertunning , with good residual plot result presented and high accuracy (R²) as 98.13%
- Feature importance analysis revealed that crop type and geographic location dominate yield prediction, while environmental variables provide secondary but consistent contributions. Feature importance estimates are highly stable across repeated subsampling.
- While this limits interpretability of climate effects, it highlights the strong structural determinants of agricultural productivity. A secondary model excluding crop and location variables was therefore considered to assess the isolated impact of environmental factors.



## References
- *World Food Program USA*. (2024, April 29). *How Climate Change Is Causing World Hunger*.  https://wfpusa.org/news/how-climate-change-is-causing-world-hunger/