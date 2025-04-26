# Mental Health Prediction from Online Survey
### Team Members: Phoebe Gao, Yina Liang Li, Carol Wang, Yuri Xu, and Qian Zhao.


## Project Overview


## Data Preprocessing and EDA
This initial phase focuses on cleaning, preprocessing, and exploring the mental health survey dataset. It prepares the data for the modeling phase by addressing the missing values, encoding categorical variables, and visualizing key insights. 

**The preprocessing steps are as follows:**

1. Dropped irrelevant columns: These are the columns named 'comments', 'timestamp', 'state', 'no_employees', and 'anonymity'.
  
2. Handled missing values: Filled the 'self_employed' variable missing values with the mode value, 'No'.

3. Cleaned age values: Filtered the dataset to include a reasonable range of age values from 18 to 100 years.

4. Standardized the gender entries: Consolidated all gender entries into 'Male', 'Female', and 'Other'.

5. Dropped duplicated rows: There are only 4 duplicated rows.

6. Encoding strategy:
    - If unique categories in the variable are less than or equal to 5, apply one-hot encoding.
    - If unique categories in the variables are more than 5, apply label encoding. 


**EDA:**

- Visualized distribution of age, gender, and treatment seeking behavior.
- Identified the imbalances in predictors using histograms.

The key insights tells us that we have a male dominant sample, the majority of the participants/respondents are from the United States, there are limited access to mental health benefits in workplaces, and mental health openness is still very low in the interview settings.
