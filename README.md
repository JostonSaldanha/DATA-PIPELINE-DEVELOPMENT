# DATA-PIPELINE-DEVELOPMENT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JOSTON SALDANHA

*INTERN ID*: CT4MOII

*DOMAIN*: DATA SCIENCE

*DURATION*: 16 WEEKS

*MENTOR*: NEELA SANTOSH

## Detailed description of the task

### Objective

The primary goal of this project is to build an end-to-end machine learning data pipeline for predicting diamond prices based on various attributes. The task involves preprocessing the dataset, transforming it into a usable format, and creating a pipeline for machine learning models. The final preprocessed dataset ensures data consistency, scalability, and better model performance.


### Tools and Libraries Used:

#### Pandas:

Purpose: Pandas is a powerful Python library primarily used for data manipulation and analysis. It simplifies tasks such as reading data, handling missing values, and performing exploratory data analysis (EDA).

Usage in the Task:
Used to load the dataset (train.csv) into a DataFrame.Columns were selected and manipulated using Pandas' built-in functions (select_dtypes for differentiating categorical and numerical features, drop for removing unnecessary columns).

Why Pandas?: 
Pandas offers an intuitive API for handling tabular data and integrates seamlessly with other Python libraries like NumPy, making it a preferred choice for data preprocessing.


#### Scikit-learn:

Purpose: Scikit-learn is a widely used machine learning library in Python. It provides tools for data preprocessing, model building, evaluation, and pipelines for streamlining workflows.

Usage in the Task:
SimpleImputer: Handles missing values in both numerical and categorical data. Missing numerical values are replaced with the mean (default strategy), while missing categorical values are replaced with the most frequent value.
StandardScaler: Standardizes numerical features by scaling them to have a mean of 0 and a standard deviation of 1.
OrdinalEncoder: Converts categorical values into numerical values based on predefined orderings (e.g., quality of diamonds in cut_categories).
Pipeline: A Scikit-learn feature that combines multiple preprocessing steps into a single, reusable workflow for numerical and categorical data.
ColumnTransformer: Allows applying different preprocessing steps to different subsets of features (e.g., separate pipelines for numerical and categorical columns).
Train-test Split: Splits the data into training and testing subsets to evaluate the pipeline.

Why Scikit-learn?: 
Its modular and flexible design enables efficient data preprocessing, making it ideal for tasks involving transformations and machine learning.


### Editor/Platform Used

VS Code (Visual Studio Code)

Jupyter Notebook


### Applications of the Task

#### 1.Real-world Applications:

Retail Analytics:
Predict product prices or demand trends using cleaned and transformed data.

E-commerce:
Analyze customer behavior to optimize pricing, identify product quality preferences, and enhance recommendation systems.

Healthcare:
Preprocess patient data for predictive modeling, such as predicting disease progression based on clinical features.

Finance:
Analyze financial records for credit scoring, fraud detection, and price predictions.

#### 2.Machine Learning Workflow:
The data preprocessing pipeline is critical in any machine learning project. This task creates a reusable framework for handling raw data, ensuring:

Clean data: By handling missing values.

Consistent data: By scaling and encoding features for compatibility with machine learning algorithms.

Efficient workflows: By automating repetitive preprocessing steps.

#### 3.Scalable Data Processing:

In production systems, pipelines like these can be scaled up to handle large datasets or integrated into data pipelines with tools like Apache Airflow.


### Steps performed in the task:

#### 1.Extract 

Extracting Data from the Source
The Extract phase involves retrieving the raw data from the source (in this case, a CSV file) and preparing it for the next steps.

a. Loading the Data:

data = pd.read_csv('./train.csv')

b. Defining the Features and Target:

X = data.drop(labels=["id", "price"], axis=1)
y = data["price"]

X: The features (id and price columns are dropped as id is not useful for model training and price is the target).
y: The target variable (price) is separated for supervised learning.

#### 2.Transform: Data Preprocessing and Transformation

The Transform phase deals with cleaning, modifying, and converting the data into the required format for analysis or machine learning models. This step is performed by applying preprocessing techniques like handling missing values, scaling numerical features, and encoding categorical variables.

a. Identifying Categorical and Numerical Columns

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

Categorical columns (cat_cols) are identified based on their data type (object).
Numerical columns (num_cols) are the ones with numeric data types (e.g., int or float).

b. Preprocessing Pipelines

Numerical Pipeline:

num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer()), 
        ("scaler", StandardScaler())
    ]
)

Imputation: Missing values in numerical columns are handled by the SimpleImputer() (default strategy is mean imputation).
Scaling: Numerical features are standardized using StandardScaler(), which transforms them to have a mean of 0 and standard deviation of 1.


Categorical Pipeline:

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinalencoder", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
    ]
)

Imputation: Missing categorical values are filled using the most frequent value within each column (SimpleImputer(strategy="most_frequent")).
Encoding: Categorical values are transformed into numerical values using OrdinalEncoder(), and specific categories are predefined (e.g., for cut, color, and clarity).

c. Combining Pipelines Using ColumnTransformer

preprocessor = ColumnTransformer(
    [
        ("num_pipeline", num_pipeline, num_cols),
        ("cat_pipeline", cat_pipeline, cat_cols)
    ]
)
The ColumnTransformer applies the respective pipelines to the appropriate columns:
Numerical columns (num_cols) are processed through the numerical pipeline.
Categorical columns (cat_cols) are processed through the categorical pipeline.

d. Splitting the Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
The data is split into training and testing datasets (70% for training, 30% for testing). This ensures that the model is trained on one subset and evaluated on another.

e. Applying the Preprocessing Pipeline

X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

Training Data: The fit_transform method applies the transformation steps (imputation, scaling, encoding) to the training data (X_train).
Testing Data: The transform method applies the same transformations to the test data (X_test) to ensure consistency.


#### 3. Load: Storing or Using Transformed Data

In the Load phase, the transformed data is either saved to a database or file, or used directly for machine learning tasks.

The transformed X_train and X_test datasets are now ready to be fed into machine learning models for training and evaluation. In this case, they are returned as Pandas DataFrames with appropriately transformed features.

X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_n)


## Output

![Image](https://github.com/user-attachments/assets/5ba68d87-4aff-488f-a18c-5e40fc3565e9)
![Image](https://github.com/user-attachments/assets/61e7a755-8162-4a76-b17a-ee0dcfc9cf0a)
![Image](https://github.com/user-attachments/assets/d621167c-1023-4c74-8216-53e2e95cb0ba)