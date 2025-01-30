import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_columns(df, columns_to_scale):
    """
    Scales specific columns in a DataFrame to the range [0, 1].
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_scale (list): List of column names to scale.
    
    Returns:
        pd.DataFrame: A DataFrame with specified columns scaled between 0 and 1.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_scaled = df.copy()
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Apply scaling to the selected columns
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    
    return df_scaled


def create_whole_story_df(df):

    # level 2
    df['subscription_type_location'] = df['subscription_type'] + '_' + df['location']
    df['subscription_type_company_size'] = df['subscription_type'] + '_' + df['company_size']
    df['subscription_type_industry'] = df['subscription_type'] + '_' + df['industry']
    df['industry_location'] = df['industry'] + '_' + df['location']
    df['industry_company_size'] = df['industry'] + '_' + df['company_size']
    df['location_company_size'] = df['location']  + '_' + df['company_size']

    # level 3
    df['location_company_size_industry'] = df['location']  + '_' + df['company_size'] + '_' + df['industry']
    df['location_company_size_industry_subscription_type'] = df['location']  + '_' + df['company_size'] + '_' + df['industry'] +df['subscription_type']


    # the complete story df
    a = df.groupby(['industry_location', 'renewed']).size().reset_index(name='count').rename(columns={'industry_location': 'indicator'})
    b = df.groupby(['industry_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'industry_company_size': 'indicator'})
    c = df.groupby(['location_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'location_company_size': 'indicator'})
    d = df.groupby(['subscription_type_location', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_location': 'indicator'})
    e = df.groupby(['subscription_type_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_company_size': 'indicator'})
    f = df.groupby(['subscription_type_industry', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_industry': 'indicator'})
    g = df.groupby(['subscription_type', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type': 'indicator'})
    h = df.groupby(['industry', 'renewed']).size().reset_index(name='count').rename(columns={'industry': 'indicator'})
    i = df.groupby(['company_size', 'renewed']).size().reset_index(name='count').rename(columns={'company_size': 'indicator'})
    j = df.groupby(['location', 'renewed']).size().reset_index(name='count').rename(columns={'location': 'indicator'})
    complete_story = pd.concat([a,b,c,d,e,f,g,h,i,j])

    # Group by  indicator and aggregate the counts for True and False
    summary = (
        complete_story.groupby(["indicator", "renewed"])["count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["n"] = summary.sum(axis=1, numeric_only=True)
    # Calculate the ratio of False
    summary["ratio_true"] = summary[True] / (summary[False] + summary[True])
    summary['depth'] = summary['indicator'].str.count('_') +1

    # Summarize into a single row (if desired, you can keep the detailed table)
    whole_story = summary[["indicator", "ratio_true","n","depth"]].sort_values(by='ratio_true', ascending=False)
    return whole_story


def plot_2_time_series(title,df1, date_column1, indicator_1,label_1,df2, date_column2, indicator_2,label_2 ):
    ''' 
    Plot 2 time series
    '''
    # Sample time series data
    # Plot the two time series
    plt.figure(figsize=(10, 6))
    plt.plot(df1[date_column1], df1[indicator_1], label=label_1, marker='o')
    plt.plot(df2[date_column2], df2[indicator_2], label=label_2, marker='x')

    # Add labels, legend, and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()
    

def plot_1_time_series_with_line(df1, date_column1, indicator_1,label_1, line_value ):
    ''' 
    Plot 2 time series
    '''
    # Sample time series data
    # Plot the two time series
    plt.figure(figsize=(10, 6))
    plt.plot(df1[date_column1], df1[indicator_1], label=label_1, marker='o')

    plt.axhline(y=line_value, color="red", linestyle="--", linewidth=2, label="Threshold")
    # Add labels, legend, and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Economic time series')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()


def preprocess_and_logistic(df2, target_column):
    ''' 
    Logistic model pipeline
    '''
    df = df2.copy()
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Preprocessing for categorical variables
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create the pipeline with Logistic Regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Hyperparameter tuning for Logistic Regression
    param_grid = {
        'classifier__C': [0.000001,0.00001,0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return best_model, grid_search

def extract_feature_importance(best_model, X):
    """
    Extracts feature importance from a Logistic Regression model in a pipeline.

    Parameters:
    - best_model: The fitted model pipeline from GridSearchCV.
    - X: The input DataFrame used for training (to get feature names).

    Returns:
    - A DataFrame with features and their importance values.
    """
    # Get the OneHotEncoder step from the pipeline
    preprocessor = best_model.named_steps['preprocessor']
    encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']

    # Get the original feature names for categorical variables
    encoded_feature_names = encoder.get_feature_names_out(X.select_dtypes(include=['object', 'category']).columns)

    # Combine numerical and encoded feature names
    feature_names = list(encoded_feature_names)

    # Extract coefficients from the logistic regression model
    classifier = best_model.named_steps['classifier']
    coefficients = classifier.coef_.flatten()

    # Create a DataFrame with feature names and their coefficients
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort by absolute coefficient value
    feature_importance_df['Absolute Coefficient'] = feature_importance_df['Coefficient'].abs()
    feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)

    return feature_importance_df

def add_avg_indicator_value_given_dates(custom_date1,custom_date2,df2):
    # Define the custom dates
    df = df2.copy()
    # Convert the custom dates to datetime
    custom_date1 = pd.to_datetime(custom_date1)
    custom_date2 = pd.to_datetime(custom_date2)

    # Ensure the 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Calculate means for the three ranges
    mean_before_date1 = df[df['date'] < custom_date1]['renewed_ratio'].mean()
    mean_between_dates = df[(df['date'] >= custom_date1) & (df['date'] < custom_date2)]['renewed_ratio'].mean()
    mean_after_date2 = df[df['date'] >= custom_date2]['renewed_ratio'].mean()

    # Add the 'analisis' column based on the ranges
    df['analisis'] = df['date'].apply(
        lambda x: mean_before_date1 if x < custom_date1 else (
            mean_between_dates if x < custom_date2 else mean_after_date2
        )
    )
    return df


def get_complete_story(df):
    # the complete story df
    a = df.groupby(['industry_location', 'renewed']).size().reset_index(name='count').rename(columns={'industry_location': 'indicator'})
    b = df.groupby(['industry_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'industry_company_size': 'indicator'})
    c = df.groupby(['location_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'location_company_size': 'indicator'})
    d = df.groupby(['subscription_type_location', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_location': 'indicator'})
    e = df.groupby(['subscription_type_company_size', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_company_size': 'indicator'})
    f = df.groupby(['subscription_type_industry', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type_industry': 'indicator'})
    g = df.groupby(['subscription_type', 'renewed']).size().reset_index(name='count').rename(columns={'subscription_type': 'indicator'})
    h = df.groupby(['industry', 'renewed']).size().reset_index(name='count').rename(columns={'industry': 'indicator'})
    i = df.groupby(['company_size', 'renewed']).size().reset_index(name='count').rename(columns={'company_size': 'indicator'})
    j = df.groupby(['location', 'renewed']).size().reset_index(name='count').rename(columns={'location': 'indicator'})
    complete_story = pd.concat([a,b,c,d,e,f,g,h,i,j])

    # Group by  indicator and aggregate the counts for True and False
    summary = (
        complete_story.groupby(["indicator", "renewed"])["count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["n"] = summary.sum(axis=1, numeric_only=True)
    # Calculate the ratio of False
    summary["ratio_true"] = summary[True] / (summary[False] + summary[True])
    summary['depth'] = summary['indicator'].str.count('_') +1

    # Summarize into a single row (if desired, you can keep the detailed table)
    #whole_story = summary[["indicator", "ratio_true","n","depth"]].sort_values(by='ratio_true', ascending=False)
    return summary

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def create_dinamic_table(df,title):

    # Get unique years for animation frames
    years = sorted(df["year"].unique())

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')

    # Function to update the table in each frame (one year at a time)
    def update(frame):
        ax.clear()
        ax.axis('tight')
        ax.axis('off')

        # Filter data for the current year
        year = years[frame]
        df_subset = df[df["year"] == year]

        # Create table and display in figure
        table = ax.table(cellText=df_subset.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2, 3])

        # Add title to indicate the year being shown
        ax.set_title(f"{title} {year}", fontsize=12, fontweight="bold")

    # Store animation object to prevent garbage collection
    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=1500, repeat=True)
    plt.close(fig)
    # Display animation in Jupyter Notebook
    return HTML(ani.to_jshtml())