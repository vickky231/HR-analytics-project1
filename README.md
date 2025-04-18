
Importing Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting Seaborn style
sns.set(style="whitegrid")

Loading Dataset

df = pd.read_csv('HR_Analytics.csv')

df.head

<bound method NDFrame.head of        EmpID  Age AgeGroup Attrition     BusinessTravel  DailyRate  \
0      RM297   18    18-25       Yes      Travel_Rarely        230   
1      RM302   18    18-25        No      Travel_Rarely        812   
2      RM458   18    18-25       Yes  Travel_Frequently       1306   
3      RM728   18    18-25        No         Non-Travel        287   
4      RM829   18    18-25       Yes         Non-Travel        247   
...      ...  ...      ...       ...                ...        ...   
1475   RM412   60      55+        No      Travel_Rarely        422   
1476   RM428   60      55+        No  Travel_Frequently       1499   
1477   RM537   60      55+        No      Travel_Rarely       1179   
1478   RM880   60      55+        No      Travel_Rarely        696   
1479  RM1210   60      55+        No      Travel_Rarely        370   

                  Department  DistanceFromHome  Education EducationField  ...  \
0     Research & Development                 3          3  Life Sciences  ...   
1                      Sales                10          3        Medical  ...   
2                      Sales                 5          3      Marketing  ...   
3     Research & Development                 5          2  Life Sciences  ...   
4     Research & Development                 8          1        Medical  ...   
...                      ...               ...        ...            ...  ...   
1475  Research & Development                 7          3  Life Sciences  ...   
1476                   Sales                28          3      Marketing  ...   
1477                   Sales                16          4      Marketing  ...   
1478                   Sales                 7          4      Marketing  ...   
1479  Research & Development                 1          4        Medical  ...   

      RelationshipSatisfaction  StandardHours  StockOptionLevel  \
0                            3             80                 0   
1                            1             80                 0   
2                            4             80                 0   
3                            4             80                 0   
4                            4             80                 0   
...                        ...            ...               ...   
1475                         4             80                 0   
1476                         4             80                 0   
1477                         4             80                 0   
1478                         2             80                 1   
1479                         3             80                 1   

     TotalWorkingYears  TrainingTimesLastYear  WorkLifeBalance  \
0                    0                      2                3   
1                    0                      2                3   
2                    0                      3                3   
3                    0                      2                3   
4                    0                      0                3   
...                ...                    ...              ...   
1475                33                      5                1   
1476                22                      5                4   
1477                10                      1                3   
1478                12                      3                3   
1479                19                      2                4   

      YearsAtCompany YearsInCurrentRole  YearsSinceLastPromotion  \
0                  0                  0                        0   
1                  0                  0                        0   
2                  0                  0                        0   
3                  0                  0                        0   
4                  0                  0                        0   
...              ...                ...                      ...   
1475              29                  8                       11   
1476              18                 13                       13   
1477               2                  2                        2   
1478              11                  7                        1   
1479               1                  0                        0   

     YearsWithCurrManager  
0                     0.0  
1                     0.0  
2                     0.0  
3                     0.0  
4                     0.0  
...                   ...  
1475                 10.0  
1476                 11.0  
1477                  2.0  
1478                  9.0  
1479                  0.0  

[1480 rows x 38 columns]>

print(df.shape)
print(df.info())

(1480, 38)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1480 entries, 0 to 1479
Data columns (total 38 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   EmpID                     1480 non-null   object 
 1   Age                       1480 non-null   int64  
 2   AgeGroup                  1480 non-null   object 
 3   Attrition                 1480 non-null   object 
 4   BusinessTravel            1480 non-null   object 
 5   DailyRate                 1480 non-null   int64  
 6   Department                1480 non-null   object 
 7   DistanceFromHome          1480 non-null   int64  
 8   Education                 1480 non-null   int64  
 9   EducationField            1480 non-null   object 
 10  EmployeeCount             1480 non-null   int64  
 11  EmployeeNumber            1480 non-null   int64  
 12  EnvironmentSatisfaction   1480 non-null   int64  
 13  Gender                    1480 non-null   object 
 14  HourlyRate                1480 non-null   int64  
 15  JobInvolvement            1480 non-null   int64  
 16  JobLevel                  1480 non-null   int64  
 17  JobRole                   1480 non-null   object 
 18  JobSatisfaction           1480 non-null   int64  
 19  MaritalStatus             1480 non-null   object 
 20  MonthlyIncome             1480 non-null   int64  
 21  SalarySlab                1480 non-null   object 
 22  MonthlyRate               1480 non-null   int64  
 23  NumCompaniesWorked        1480 non-null   int64  
 24  Over18                    1480 non-null   object 
 25  OverTime                  1480 non-null   object 
 26  PercentSalaryHike         1480 non-null   int64  
 27  PerformanceRating         1480 non-null   int64  
 28  RelationshipSatisfaction  1480 non-null   int64  
 29  StandardHours             1480 non-null   int64  
 30  StockOptionLevel          1480 non-null   int64  
 31  TotalWorkingYears         1480 non-null   int64  
 32  TrainingTimesLastYear     1480 non-null   int64  
 33  WorkLifeBalance           1480 non-null   int64  
 34  YearsAtCompany            1480 non-null   int64  
 35  YearsInCurrentRole        1480 non-null   int64  
 36  YearsSinceLastPromotion   1480 non-null   int64  
 37  YearsWithCurrManager      1423 non-null   float64
dtypes: float64(1), int64(25), object(12)
memory usage: 439.5+ KB
None

Data Preprocessing

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Encoding categorical variables if necessary
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Example of creating a new feature
data['TotalWorkingYears_bin'] = pd.cut(data['TotalWorkingYears'], bins=[0, 5, 10, 15, 20, np.inf], labels=['0-5', '5-10', '10-15', '15-20', '20+'])

# Display summary of the processed data
data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1480 entries, 0 to 1479
Data columns (total 39 columns):
 #   Column                    Non-Null Count  Dtype   
---  ------                    --------------  -----   
 0   EmpID                     1480 non-null   object  
 1   Age                       1480 non-null   int64   
 2   AgeGroup                  1480 non-null   object  
 3   Attrition                 1480 non-null   object  
 4   BusinessTravel            1480 non-null   object  
 5   DailyRate                 1480 non-null   int64   
 6   Department                1480 non-null   object  
 7   DistanceFromHome          1480 non-null   int64   
 8   Education                 1480 non-null   int64   
 9   EducationField            1480 non-null   object  
 10  EmployeeCount             1480 non-null   int64   
 11  EmployeeNumber            1480 non-null   int64   
 12  EnvironmentSatisfaction   1480 non-null   int64   
 13  Gender                    1480 non-null   int64   
 14  HourlyRate                1480 non-null   int64   
 15  JobInvolvement            1480 non-null   int64   
 16  JobLevel                  1480 non-null   int64   
 17  JobRole                   1480 non-null   object  
 18  JobSatisfaction           1480 non-null   int64   
 19  MaritalStatus             1480 non-null   object  
 20  MonthlyIncome             1480 non-null   int64   
 21  SalarySlab                1480 non-null   object  
 22  MonthlyRate               1480 non-null   int64   
 23  NumCompaniesWorked        1480 non-null   int64   
 24  Over18                    1480 non-null   object  
 25  OverTime                  1480 non-null   object  
 26  PercentSalaryHike         1480 non-null   int64   
 27  PerformanceRating         1480 non-null   int64   
 28  RelationshipSatisfaction  1480 non-null   int64   
 29  StandardHours             1480 non-null   int64   
 30  StockOptionLevel          1480 non-null   int64   
 31  TotalWorkingYears         1480 non-null   int64   
 32  TrainingTimesLastYear     1480 non-null   int64   
 33  WorkLifeBalance           1480 non-null   int64   
 34  YearsAtCompany            1480 non-null   int64   
 35  YearsInCurrentRole        1480 non-null   int64   
 36  YearsSinceLastPromotion   1480 non-null   int64   
 37  YearsWithCurrManager      1480 non-null   float64 
 38  TotalWorkingYears_bin     1469 non-null   category
dtypes: category(1), float64(1), int64(26), object(11)
memory usage: 441.2+ KB

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\431514447.py:2: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  data.fillna(method='ffill', inplace=True)

# Drop duplicates if any
df = df.drop_duplicates()
print(df.shape)

(1473, 38)

print(df.columns)

Index(['EmpID', 'Age', 'AgeGroup', 'Attrition', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender',
       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
       'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'SalarySlab',
       'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager'],
      dtype='object')

# Describe the dataset
print(df.describe().round(2))

           Age  DailyRate  DistanceFromHome  Education  EmployeeCount  \
count  1473.00    1473.00           1473.00    1473.00         1473.0   
mean     36.92     802.66              9.20       2.91            1.0   
std       9.13     403.25              8.11       1.02            0.0   
min      18.00     102.00              1.00       1.00            1.0   
25%      30.00     465.00              2.00       2.00            1.0   
50%      36.00     802.00              7.00       3.00            1.0   
75%      43.00    1157.00             14.00       4.00            1.0   
max      60.00    1499.00             29.00       5.00            1.0   

       EmployeeNumber  EnvironmentSatisfaction  HourlyRate  JobInvolvement  \
count         1473.00                  1473.00     1473.00         1473.00   
mean          1026.98                     2.72       65.83            2.73   
std            603.22                     1.09       20.35            0.71   
min              1.00                     1.00       30.00            1.00   
25%            492.00                     2.00       48.00            2.00   
50%           1024.00                     3.00       66.00            3.00   
75%           1558.00                     4.00       83.00            3.00   
max           2068.00                     4.00      100.00            4.00   

       JobLevel  ...  RelationshipSatisfaction  StandardHours  \
count   1473.00  ...                   1473.00         1473.0   
mean       2.06  ...                      2.71           80.0   
std        1.11  ...                      1.08            0.0   
min        1.00  ...                      1.00           80.0   
25%        1.00  ...                      2.00           80.0   
50%        2.00  ...                      3.00           80.0   
75%        3.00  ...                      4.00           80.0   
max        5.00  ...                      4.00           80.0   

       StockOptionLevel  TotalWorkingYears  TrainingTimesLastYear  \
count           1473.00            1473.00                1473.00   
mean               0.79              11.28                   2.80   
std                0.85               7.78                   1.29   
min                0.00               0.00                   0.00   
25%                0.00               6.00                   2.00   
50%                1.00              10.00                   3.00   
75%                1.00              15.00                   3.00   
max                3.00              40.00                   6.00   

       WorkLifeBalance  YearsAtCompany  YearsInCurrentRole  \
count          1473.00         1473.00             1473.00   
mean              2.76            7.00                4.23   
std               0.71            6.12                3.62   
min               1.00            0.00                0.00   
25%               2.00            3.00                2.00   
50%               3.00            5.00                3.00   
75%               3.00            9.00                7.00   
max               4.00           40.00               18.00   

       YearsSinceLastPromotion  YearsWithCurrManager  
count                  1473.00               1416.00  
mean                      2.18                  4.12  
std                       3.22                  3.56  
min                       0.00                  0.00  
25%                       0.00                  2.00  
50%                       1.00                  3.00  
75%                       3.00                  7.00  
max                      15.00                 17.00  

[8 rows x 26 columns]

Visualizing

# Creating a pie chart for gender distribution
gender_counts = data['Gender'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()

No description has been provided for this image

plt.figure(figsize=(10, 6))
avg_salary = data.groupby('Department')['MonthlyIncome'].mean().sort_values()

sns.barplot(x=avg_salary.values, y=avg_salary.index, palette='coolwarm')
plt.title('Average Monthly Income by Department')
plt.xlabel('Average Monthly Income')
plt.ylabel('Department')
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\3723177616.py:4: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=avg_salary.values, y=avg_salary.index, palette='coolwarm')

No description has been provided for this image

plt.figure(figsize=(10, 6))
sns.histplot(data, x='Age', hue='Department', multiple='stack', palette='Set2')
plt.title('Age Distribution by Department')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

No description has been provided for this image

# Distribution of Job Satisfaction
plt.figure(figsize=(10, 6))
sns.histplot(df['JobSatisfaction'], bins=30, kde=True, color='blue')
plt.title('Distribution of Job Satisfaction Level')
plt.xlabel('Job Satisfaction Level')
plt.ylabel('Frequency')
plt.show()

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a boxplot with enhanced aesthetics
sns.boxplot(x='Attrition', y='JobSatisfaction', data=df, palette='viridis')

# Add a pointplot to show median values for each group
sns.pointplot(x='Attrition', y='JobSatisfaction', data=df, 
              estimator=np.median, color='black', markers='D', linestyles='--')

# Set plot title and labels with improved descriptions
plt.title('Comparison of Job Satisfaction Between Employees Who Left and Stayed', fontsize=16)
plt.xlabel('Employee Attrition Status (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Job Satisfaction Level (1 = Low, 4 = High)', fontsize=14)

# Adding gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\3436761315.py:7: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='Attrition', y='JobSatisfaction', data=df, palette='viridis')

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a histogram with KDE (Kernel Density Estimate)
sns.histplot(df['PerformanceRating'], bins=15, kde=True, color='green', edgecolor='black', linewidth=1.2)

# Add a vertical line for the mean of Performance Ratings
mean_rating = df['PerformanceRating'].mean()
plt.axvline(mean_rating, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rating:.2f}')

# Set plot title and labels with improved descriptions
plt.title('Distribution of Performance Ratings Among Employees', fontsize=16)
plt.xlabel('Performance Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Add a legend to describe the mean line
plt.legend()

# Show the plot
plt.show()

No description has been provided for this image

# Analyzing Salary Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='SalarySlab', data=df, palette='muted')
plt.title('Distribution of Salary Levels')
plt.xlabel('Salary Slab')
plt.ylabel('Count')
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\2488374748.py:3: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x='SalarySlab', data=df, palette='muted')

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a count plot with improved aesthetics
sns.countplot(x='SalarySlab', hue='Attrition', data=df, palette='viridis', edgecolor='black', linewidth=1.2)

# Add percentage labels on top of the bars
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), 
                       ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                       textcoords='offset points')

# Set plot title and labels with improved descriptions
plt.title('Impact of Salary Slabs on Employee Attrition', fontsize=16)
plt.xlabel('Salary Slab', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)

# Adjust legend title for clarity
plt.legend(title='Attrition (0 = No, 1 = Yes)', fontsize=12)

# Show the plot
plt.show()

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a histogram with KDE (Kernel Density Estimate)
sns.histplot(df['TotalWorkingYears'], bins=20, kde=True, color='orange', edgecolor='black', linewidth=1.2)

# Add a vertical line for the mean of Total Working Years
mean_working_years = df['TotalWorkingYears'].mean()
plt.axvline(mean_working_years, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_working_years:.1f} years')

# Add a vertical line for the median of Total Working Years
median_working_years = df['TotalWorkingYears'].median()
plt.axvline(median_working_years, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_working_years:.1f} years')

# Set plot title and labels with improved descriptions
plt.title('Distribution of Total Working Years Among Employees', fontsize=16)
plt.xlabel('Total Working Years', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Add a legend to explain the mean and median lines
plt.legend()

# Show the plot
plt.show()

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a histogram with KDE (Kernel Density Estimate)
sns.histplot(df['YearsAtCompany'], bins=15, kde=True, color='purple', edgecolor='black', linewidth=1.2)

# Add a vertical line for the mean of Years At Company
mean_years_at_company = df['YearsAtCompany'].mean()
plt.axvline(mean_years_at_company, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_years_at_company:.1f} years')

# Add a vertical line for the median of Years At Company
median_years_at_company = df['YearsAtCompany'].median()
plt.axvline(median_years_at_company, color='red', linestyle='-.', linewidth=2, label=f'Median: {median_years_at_company:.1f} years')

# Set plot title and labels with improved descriptions
plt.title('Distribution of Years at Company Among Employees', fontsize=16)
plt.xlabel('Years at Company', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Add a legend to explain the mean and median lines
plt.legend()

# Show the plot
plt.show()

No description has been provided for this image

plt.figure(figsize=(12, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a boxplot with enhanced aesthetics
sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, palette='viridis', linewidth=1.2)

# Add a pointplot to show median values for each group
sns.pointplot(x='Attrition', y='YearsAtCompany', data=df, 
              estimator=np.median, color='black', markers='D', linestyles='--', dodge=True)

# Set plot title and labels with improved descriptions
plt.title('Comparison of Years at Company Between Employees Who Left and Stayed', fontsize=16)
plt.xlabel('Employee Attrition Status (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Years at Company', fontsize=14)

# Adding gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\1173291785.py:7: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, palette='viridis', linewidth=1.2)

No description has been provided for this image

# Department Distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Department', data=df, palette='viridis')
plt.title('Distribution of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\1514300416.py:3: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x='Department', data=df, palette='viridis')

No description has been provided for this image

plt.figure(figsize=(14, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a countplot with enhanced aesthetics
sns.countplot(x='Department', data=df, palette='magma', edgecolor='black', linewidth=1.2)

# Add percentage labels on top of the bars
total = len(df['Department'])
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height} ({height/total:.1%})', (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                       textcoords='offset points')

# Set plot title and labels with improved descriptions
plt.title('Distribution of Employees by Department', fontsize=18)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)

# Adjust x-ticks for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)

# Show the plot
plt.show()

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\2109771471.py:7: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(x='Department', data=df, palette='magma', edgecolor='black', linewidth=1.2)

No description has been provided for this image

plt.figure(figsize=(14, 8))

# Set the background style
sns.set(style="whitegrid")

# Create a countplot with enhanced aesthetics
sns.countplot(x='Department', hue='Attrition', data=df, palette='coolwarm', edgecolor='black', linewidth=1.2)

# Add percentage labels on top of the bars
total_counts = df['Department'].value_counts().sort_index()
for department in df['Department'].unique():
    department_total = total_counts[department]
    for p in plt.gca().patches:
        if p.get_x() == plt.gca().get_xticks()[list(df['Department'].unique()).index(department)]:
            height = p.get_height()
            plt.gca().annotate(f'{height} ({height/department_total:.1%})',
                               (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                               textcoords='offset points')

# Set plot title and labels with improved descriptions
plt.title('Employee Attrition Across Departments', fontsize=18)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Number of Employees', fontsize=14)

# Adjust x-ticks for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)

# Add a legend with a clear title
plt.legend(title='Attrition (0 = No, 1 = Yes)', fontsize=12)

# Show the plot
plt.show()

No description has been provided for this image
Modeling

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Convert 'Attrition' column to binary values if needed
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Verify the conversion
print(df['Attrition'].value_counts())

Series([], Name: count, dtype: int64)

C:\Users\aryan\AppData\Local\Temp\ipykernel_18156\2298730866.py:2: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Feature selection
features = df[['JobSatisfaction', 'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome']]
target = df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

  LogisticRegression
?
i

LogisticRegression(max_iter=1000)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

C:\Users\aryan\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
C:\Users\aryan\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
C:\Users\aryan\anaconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))

# Print model evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

Accuracy: 85.76%
Confusion Matrix:
[[253   0]
 [ 42   0]]
Classification Report:
              precision    recall  f1-score   support

           0       0.86      1.00      0.92       253
           1       0.00      0.00      0.00        42

    accuracy                           0.86       295
   macro avg       0.43      0.50      0.46       295
weighted avg       0.74      0.86      0.79       295

conf_matrix = np.array([[253, 0], [42, 0]])

# Improved Confusion Matrix Visualization
plt.figure(figsize=(10, 8))

# Create a heatmap with enhanced aesthetics
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"size": 14, "weight": 'bold'}, 
            linewidths=.5, linecolor='black')

# Set plot title and labels with improved descriptions
plt.title('Confusion Matrix\nAccuracy: 85.76%', fontsize=18)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('Actual Labels', fontsize=14)

# Set tick labels for readability
plt.xticks(ticks=[0.5, 1.5], labels=['Class 0', 'Class 1'], fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=['Class 0', 'Class 1'], fontsize=12)

# Add text annotations for precision, recall, and F1-score
plt.text(0.5, -0.2, 'Precision: 0.86\nRecall: 1.00\nF1-Score: 0.92', 
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.text(1.5, -0.2, 'Precision: 0.00\nRecall: 0.00\nF1-Score: 0.00', 
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Show the plot
plt.show()

No description has been provided for this image

 

