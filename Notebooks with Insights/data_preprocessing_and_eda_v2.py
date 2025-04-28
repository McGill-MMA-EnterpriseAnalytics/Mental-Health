"""#Data Preprocessing"""

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Load the dataset
df = pd.read_csv('survey.csv')

#display all columns
pd.set_option('display.max_columns', None)

#show the first 5 rows of the dataset
df.head()

#show general information
df.info()

"""*Note: As we can observe from the data information, all the columns except for 'Age' are categorical variables. This makes it unfeasible to perform correlation analysis or plotting a correlation heatmap. Having mostly categorical columns, makes correlation values undefined, which means a blank plot if plotted.*"""

#Check for missing values for each column/variable
df.isnull().sum()

#describe the numerical age column
df.describe()

#drop timestamp variable
df = df.drop(columns=['comments', 'Timestamp', 'state', 'no_employees', 'anonymity'])

"""*Note: As we can observe from the general information of the dataset, there are 27 columns, having most of the variables as categorical vairables. We might not not need some of them for the project. Therefore, we proceed to remove them. The columns are: 'TimeStamp' and 'Comments'.*

*The 'Timestamp', 'no_employees', 'anonymity' columns are being removed since it is irrelevant in this use case.*

*There are missing values in columns like 'state', 'self_employed', 'work_interfere', and 'Comments'.*

*There are missing values (41%) in the 'state' column due to the fact that it only shows the state only for participants who answered 'United States' for the country variable. Therefore, we remove the 'state column since we can visualize the survey insights by referring to the 'Country' variable.*

*The 'self_employed' variable has a few missing values (1.4%), we proceed to replace the missing values with the mode or most frequent answer, that is either 'yes' or 'no'.*

*The 'Comments' column is being removed due to highly missing values/comments for 87% of the data (1095 rows).*

*Moreover, we can also observe that the 'Age' column contains some extreme values like a maximum of 1e+11 and minimum of -1726, where it needs cleaning. Therefore, we proceed to keep only the reasonable range of age from 18 to 100.*
"""

#verify the unique values and count of the self employed column, including nulls
df['self_employed'].value_counts(dropna=False)

#fill the null values of the self employed column with 'No'
df['self_employed'] = df['self_employed'].fillna('No')

#verify unique values of the column
df['self_employed'].value_counts(dropna=False)

"""*Note: As we can observe, the 'self_employed' column's missing values are being replaced with 'No' since it has the majority of the count (mode).*"""

#clean the age column by keeping the reasonable range of age from 18 to 100
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

#updated information on dataset
df.info()

#updated age filter on dataset
df.describe()

"""*Note: As we can observe, the irrelevant columns are being removed and the 'Age' variable is being filtered to keep only the range of reasonable age from 18 to 100. However, the maximum age is 72, with a 75% of participants aging around 36 years old.*

*Now we proceed to modify the 'Gender' variable. As we can observe from the first five rows shown earlier, there are different answers besides 'female' and 'male'. That is, some participants of the survey answered 'M', 'F', 'woman', 'men', etc. Therefore, we need to unify and fix the 'Gender' variable into 'female', 'male', and 'other'.*
"""

#verify distinct gender values
df['Gender'].unique()

#Function to clean the gender variable into 'female', 'male', and 'other'
def clean_gender(gender):
    gender = str(gender).strip().lower()
    if gender in ['male', 'm', 'cis male', 'male (cis)', 'man', 'mail', 'cis man', 'malr', 'make', 'maile']:
        return 'Male'
    elif gender in ['female', 'f', 'cis female', 'woman', 'female (cis)', 'cis-female/femme', 'femake', 'femail', 'female ', 'trans female']:
        return 'Female'
    else:
        return 'Other'

# Apply the cleaning function
df['Gender'] = df['Gender'].apply(clean_gender)

#verify the results
df['Gender'].value_counts()

"""*Note: Now we have classified the 'Gender' variable into 'Male', 'Female', and 'Other', which will be easier to visualize insights later in the EDA.*

*We proceed to check if there are duplicated rows and remove them accordingly.*
"""

#check if there are duplicates
duplicated_rows = df[df.duplicated()]
print("Number of duplicated rows:", len(duplicated_rows))

#remove duplicated rows
df = df.drop_duplicates()

#verify the results
duplicated_rows = df[df.duplicated()]
print("Number of duplicated rows:", len(duplicated_rows))

"""*Note: As we have finalized the preprocessing steps, we will now proceed to the EDA stage.*

# EDA
"""

#set up a clean style
sns.set(style='whitegrid')

#Age distribution
plt.figure(figsize=(8, 8))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

"""*Note: As we can observe from the distribution plot, most respondents are between 20 to 40 years old and the distribution is slightly right-skewed.*"""

#pie chart for gender distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Gender Distribution')
plt.show()

"""*Note: around 79% of the participants are male.*"""

#Gender vs Treatment plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='treatment', data=df, palette='bright')
plt.title('Treatment Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""*Note: Males are the majority of the respondents that has sought treatment for mental health.*"""

#Top 10 country distribution plot
top_countries = df['Country'].value_counts().nlargest(10).index
plt.figure(figsize=(10, 6))
sns.countplot(data=df[df['Country'].isin(top_countries)], y='Country', order=top_countries, palette='bright', hue='Country')
plt.title('Top 10 Countries in the Survey')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

"""*Note: The majority of the participants comes from the United States followed by the UK and Canada.*"""

#family history vs treatment plot
plt.figure(figsize=(10, 6))
sns.countplot(x='family_history', hue='treatment', data=df)
plt.xlabel('Family History')
plt.ylabel('Count')
plt.title('Family History vs Treatment')
plt.show()

"""*Note: Those respondents with a family history of mental illness are significantly more likely to seek treatment.*"""

# work interfere vs treatment plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='work_interfere', hue='treatment', palette="bright")
plt.title('Work Interference vs Treatment')
plt.xlabel('Work Interference')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""*Note: Participants who report that mental health 'often' or 'sometimes' interferes with work are more likely to seek treatment.*"""

#remote work vs treatment plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='remote_work', hue='treatment', palette="bright")
plt.title('Remote Work vs Treatment')
plt.xlabel('Remote Work')
plt.ylabel('Count')
plt.show()

"""*Note: treatment rates are relatively similar between those respondents who work remotely and those who do not.*"""

#tech company vs treatment plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='tech_company', hue='treatment', palette="bright")
plt.title('Tech Company vs Treatment')
plt.xlabel('Tech Company')
plt.ylabel('Count')
plt.show()

"""*Note: it demonstrates that there are slightly more people in tech companies that sought treatment, but the difference is not huge.*"""

#benefits vs treatment plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='benefits', hue='treatment')
plt.title('Mental Health Benefits vs Treatment')
plt.xlabel('Mental Health Benefits')
plt.ylabel('Count')
plt.show()

"""*Note: Those respondents who do not have mental health benefits are mroe likely to seek treatment and the uncertainty about the benefits (those who responded 'dont know') seems to correlate with the lower treatment rates.*"""

#care options vs treatment plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='care_options', hue='treatment', palette='pastel')
plt.title('Care Options vs Treatment')
plt.xlabel('Care Options')
plt.ylabel('Count')
plt.show()

"""*Note: This plot shows that the access to care options strongly correlates with treatment. That is, respondents who reported having care options are much more likely to have sought treatment.*"""

# Function to create count plots for categorical columns vs treatment
def plot_vs_treatment(column, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=column, hue='treatment')
    plt.title(f'{column.replace("_", " ").title()} vs Treatment')
    plt.tight_layout()
    plt.show()

# supervisor and coworker vs treatment plots
eda_columns = ['supervisor', 'coworkers']

# Generate plots
for col in eda_columns:
    plot_vs_treatment(col)

"""*Note: Respondents with supportive supervisors and coworkers are more likely to seek treatment.*"""

# wellness programs & help seeking vs treatment plots
eda_columns = ['wellness_program', 'seek_help']

# Generate plots
for col in eda_columns:
    plot_vs_treatment(col)

"""*Note: the presence of wellness programs and encouragement to seek help are correlated with higher treatment rates.*

*Note: Now we proceed to add histograms to gain insights from predictor imbalances. Before encoding the variables, we will proceed to loop through all categorical predictors and insert plots to visualize class imbalances (if any)*.
"""

#show imbalance plots for categorical predictors before encoding
#exclude target variable 'treatment'
categorical_cols = df.select_dtypes(include='object').columns.drop('treatment')

n = len(categorical_cols)
n_cols = 2
n_rows = (n + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=axes[i])
    axes[i].set_title(f'Imbalance in {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""*Note: As we can observe from the histograms, it guides us in feature selection when modeling and these are the insights:*

*1. Gender: highly imbalanced with around 80% of male respondents and around 20% of female respondents.*

*2. Country: skewed only dominated by having US, UK, and Canada respondents, limiting generalizability.*

*3. Self Employed: Around 90% responding 'No', meaning that job structure might not significantly vary across the respondents.*

*4. Family History: Moderate imbalance showing that slightly more poeple without family history of having mental illness. However, there are enough variation for modeling purposes.*

*5. Work Interference: Relatively balanced, showing it as a good feature with potential predictive power.*

*6. Remote work: skewed towards aroung 80% 'No'. Meaning that there is limited remote work adoption in the sample.*

*7. Tech company: Strong imbalance towards 'No'.*

*8. Supervisor and Coworker support: most people report having some level of support, it has decent spread, which is good for modeling.*

*9. Benefits: Fairly balanced, which is a valuable variable to explore treatment behavior.*

*10. Care options: balanced, making it a potentially strong feature.*

*11. Wellness programs and Seek help policy: Both highly imbalanced, having large majority reporting no programs and most companies have no policy or employees are unaware. This demonstrates that there is lack of mental health infrastructure in workplaces reported and could influence mental health treatment behavior.*

*12. Mental and physical health consequences: strongly skewed towards 'No', meaning that most do not perceive any work related consequences for disclosing tehir mental or physical health issues.*

*13. Observed consequences: Skewed towards 'No', suggesting that stigma may not be commonly observed. But it could also be unnoticed or unreported.*

*14. Mental health interview: extremely skewed having almost all respondents saying 'No', meaning that there could be discomfort around disclosing mental health concerns in the hiring context.*

*15. Physical health interview: balanced, indicates greater comfort when discussing physical health in interviews.*

*16. Mental vs Physical health: balanced. Many respondents see mental health as equally important as physical health. *

*17. Leave policy: moderate imbalance having 'dont know' as the most frequent response. This tells us that there could be communication gap or a lack of formal policies.*

*Note: After finalizing the EDA, we proceed to handle the categorical variables by encoding them after the EDA and before the modeling phase. It will be much easier to analyze and visualize the categorical variables in their original form rather as an encoded vector. The way we are handling the categorical variables is based on the unique value counts. That is, applying one-hot encoding if the number of unique values in a variable is less than or equal to 5. And, applying label encoding if the number of unique values in a variable is more than 5.*
"""

#encoding logic
df_encoded = df.copy()
label_encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    # skip target
    if col == 'treatment':
        continue
    unique_vals = df_encoded[col].nunique()
    if unique_vals <= 5:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True, dtype=int)
        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
    else:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

#show the shape of the encoded dataset
print(df_encoded.shape)

#first rows of the encoded dataset
df_encoded.head()

#save encoded dataset as csv
df_encoded.to_csv('survey_encoded_v2.csv', index=False)
