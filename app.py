import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from scipy import stats
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import traceback
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

def visualize_missing_values(df: pd.DataFrame, column: str) -> None:
    st.subheader("📊 Missing Values Visualization")
    # Calculate missing values statistics
    total_rows = len(df)
    missing_count = df[column].isnull().sum()
    missing_percentage = (missing_count / total_rows * 100).round(2)
    
    # Create a pie chart using plotly
    fig = go.Figure(data=[go.Pie(
        labels=['Present Values', 'Missing Values'],
        values=[total_rows - missing_count, missing_count],
        hole=0.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig.update_layout(
        title=f"Missing Values Distribution for '{column}'",
        annotations=[{
            'text': f'{missing_percentage}%<br>Missing',
            'x': 0.5,
            'y': 0.5,
            'font_size': 20,
            'showarrow': False
        }]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", total_rows)
    with col2:
        st.metric("Missing Values", missing_count)
    with col3:
        st.metric("Missing Percentage", f"{missing_percentage}%")
    
    # If numeric, show distribution of non-missing values
    if pd.api.types.is_numeric_dtype(df[column]):
        st.subheader("📈 Distribution of Non-Missing Values")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            nbinsx=30,
            name="Distribution",
            marker_color='#3498db'
        ))
        fig.update_layout(
            title=f"Distribution of Values in '{column}'",
            xaxis_title=column,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced interactive Streamlit UI for handling missing values."""
    st.header("🧹 Missing Values Analysis and Treatment")
    
    # Overall missing values summary
    missing_summary = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    
    total_missing = missing_summary['Missing Count'].sum()
    
    if total_missing == 0:
        st.success("✨ No missing values in the dataset!")
        return df
    
    # Display missing values summary
    st.subheader("Missing Values Summary")
    st.dataframe(missing_summary[missing_summary['Missing Count'] > 0])
    
    # Column selection for missing values handling
    columns_with_missing = missing_summary[missing_summary['Missing Count'] > 0].index.tolist()
    selected_column = st.selectbox(
        "Select a column to handle missing values",
        columns_with_missing,
        help="Choose a column to analyze and treat missing values"
    )
    
    # Add visualization for the selected column
    visualize_missing_values(df, selected_column)
    
    # Check if the selected column is numeric or not
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_column])
    
    # Methods list based on the data type
    all_methods = [
        "Fill with Mode", 
        "Random Imputation", 
        "Fill with Custom Value", 
        "Drop Rows"
    ]
    
    if is_numeric:
        # Add methods for numeric columns
        all_methods = ["Fill with Mean", "Fill with Median"] + all_methods
    
    # Handling method selection
    st.subheader("🛠️ Missing Values Treatment")
    method = st.selectbox("Choose a method to treat missing values", all_methods)
    
    # Additional parameters based on method
    params = {}
    if method == "Fill with Custom Value":
        params['custom_value'] = st.text_input("Enter custom value")
    
    # Preview changes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("👀 Original Data Preview")
        st.dataframe(df[[selected_column]])
    
    with col2:
        st.write("✨ Preview After Treatment")
        try:
            modified_series = df[selected_column].copy()
            
            if method == "Fill with Mean" and is_numeric:
                modified_series.fillna(modified_series.mean(), inplace=True)
            elif method == "Fill with Median" and is_numeric:
                modified_series.fillna(modified_series.median(), inplace=True)
            elif method == "Fill with Mode":
                modified_series.fillna(modified_series.mode()[0], inplace=True)
            elif method == "Random Imputation":
                non_null_values = modified_series.dropna().values
                modified_series = modified_series.apply(
                    lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
                )
            elif method == "Fill with Custom Value" and params.get('custom_value'):
                modified_series.fillna(params['custom_value'], inplace=True)
            elif method == "Drop Rows":
                modified_series = modified_series.dropna()
            
            st.dataframe(modified_series)
            
        except Exception as e:
            st.error(f"Error previewing changes: {str(e)}")
    
    with col3:
        # Show statistics comparison for numeric columns
        if is_numeric:
            stats_comparison = pd.DataFrame({
                'Original': df[selected_column].describe(),
                'After Treatment': modified_series.describe()
            })
            st.write("📊 Statistics Comparison")
            st.dataframe(stats_comparison)
    
    # Apply changes
    if st.button("Apply Changes", type="primary"):
        try:
            if method == "Fill with Mean" and is_numeric:
                df[selected_column].fillna(df[selected_column].mean(), inplace=True)
            elif method == "Fill with Median" and is_numeric:
                df[selected_column].fillna(df[selected_column].median(), inplace=True)
            elif method == "Fill with Mode":
                df[selected_column].fillna(df[selected_column].mode()[0], inplace=True)
            elif method == "Random Imputation":
                non_null_values = df[selected_column].dropna().values
                df[selected_column] = df[selected_column].apply(
                    lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
                )
            elif method == "Fill with Custom Value" and params.get('custom_value'):
                df[selected_column].fillna(params['custom_value'], inplace=True)
            elif method == "Drop Rows":
                df.dropna(subset=[selected_column], inplace=True)
            
            st.success(f"✅ Successfully handled missing values in '{selected_column}'!")
            
            # Show remaining missing values
            remaining_missing = df[selected_column].isnull().sum()
            if remaining_missing > 0:
                st.warning(f"ℹ️ {remaining_missing} missing values remain in this column")
            else:
                st.success("🎉 No missing values remain in this column!")
                
        except Exception as e:
            st.error(f"Error applying changes: {str(e)}")
    
    return df



def visualize_outliers(df: pd.DataFrame, column: str, lower_bound: float, upper_bound: float):
    """Create visualizations for outlier analysis."""
    # Create box plot and histogram side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df[column],
            name=column,
            boxpoints='outliers',  # show outliers
            marker_color='rgb(7, 40, 89)',
            line_color='rgb(7, 40, 89)'
        ))
        fig.update_layout(
            title=f'Box Plot for {column}',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Histogram with bounds marked
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column],
            name='Distribution',
            nbinsx=30
        ))
        
        # Add vertical lines for bounds
        fig.add_vline(x=lower_bound, line_dash="dash", line_color="red",
                     annotation_text="Lower Bound")
        fig.add_vline(x=upper_bound, line_dash="dash", line_color="red",
                     annotation_text="Upper Bound")
        
        fig.update_layout(
            title=f'Distribution of {column} with Outlier Bounds',
            xaxis_title=column,
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.write("**Summary Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df[column].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[column].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[column].std():.2f}")
    with col4:
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        st.metric("Outliers", f"{len(outliers)}")



st.title("Data Preprocessing Project")
st.divider()

with st.sidebar:
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader(
        "Choose a dataset",
        type="csv",
        help="Supported formats: csv"
    )


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced interactive Streamlit UI for handling missing values."""
    st.header("🧹 Missing Values Analysis and Treatment")
    
    # Overall missing values summary
    missing_summary = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    
    total_missing = missing_summary['Missing Count'].sum()
    
    if total_missing == 0:
        st.success("✨ No missing values in the dataset!")
        return df
    
    # Display missing values summary
    st.subheader("Missing Values Summary")
    st.dataframe(missing_summary[missing_summary['Missing Count'] > 0])
    
    # Column selection for missing values handling
    columns_with_missing = missing_summary[missing_summary['Missing Count'] > 0].index.tolist()
    selected_column = st.selectbox(
        "Select a column to handle missing values",
        columns_with_missing,
        help="Choose a column to analyze and treat missing values"
    )
    visualize_missing_values(df, selected_column)
    # Check if the selected column is numeric or not
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_column])
    
    # Methods list based on the data type
    all_methods = [
        "Fill with Mode", 
        "Random Imputation", 
        "Fill with Custom Value", 
        "Drop Rows"
    ]
    
    if is_numeric:
        # Add methods for numeric columns
        all_methods = ["Fill with Mean", "Fill with Median"] + all_methods
    
    # Handling method selection
    st.subheader("🛠️ Missing Values Treatment")
    method = st.selectbox("Choose a method to treat missing values", all_methods)
    
    # Additional parameters based on method
    params = {}
    if method == "Fill with Custom Value":
        params['custom_value'] = st.text_input("Enter custom value")
    
    # Preview changes
    col1, col2 , col3 = st.columns(3)
    with col1:
        st.write("👀 Original Data Preview")
        # Show the entire column
        st.dataframe(df[[selected_column]])
    
    with col2:
        st.write("✨ Preview After Treatment")
        try:
            modified_series = df[selected_column].copy()
            
            if method == "Fill with Mean" and is_numeric:
                modified_series.fillna(modified_series.mean(), inplace=True)
            elif method == "Fill with Median" and is_numeric:
                modified_series.fillna(modified_series.median(), inplace=True)
            elif method == "Fill with Mode":
                modified_series.fillna(modified_series.mode()[0], inplace=True)
            elif method == "Random Imputation":
                non_null_values = modified_series.dropna().values
                modified_series = modified_series.apply(
                    lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
                )
            elif method == "Fill with Custom Value" and params.get('custom_value'):
                modified_series.fillna(params['custom_value'], inplace=True)
            elif method == "Drop Rows":
                modified_series = modified_series.dropna()
            
            st.dataframe(modified_series)
            
        except Exception as e:
            st.error(f"Error previewing changes: {str(e)}")
    
    with col3:
            # Show statistics comparison for numeric columns
            if is_numeric:
                stats_comparison = pd.DataFrame({
                    'Original': df[selected_column].describe(),
                    'After Treatment': modified_series.describe()
                })
                st.write("📊 Statistics Comparison")
                st.dataframe(stats_comparison)
    
    # Apply changes
    if st.button("Apply Changes", type="primary"):
        try:
            if method == "Fill with Mean" and is_numeric:
                df[selected_column].fillna(df[selected_column].mean(), inplace=True)
            elif method == "Fill with Median" and is_numeric:
                df[selected_column].fillna(df[selected_column].median(), inplace=True)
            elif method == "Fill with Mode":
                df[selected_column].fillna(df[selected_column].mode()[0], inplace=True)
            elif method == "Random Imputation":
                non_null_values = df[selected_column].dropna().values
                df[selected_column] = df[selected_column].apply(
                    lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
                )
            elif method == "Fill with Custom Value" and params.get('custom_value'):
                df[selected_column].fillna(params['custom_value'], inplace=True)
            elif method == "Drop Rows":
                df.dropna(subset=[selected_column], inplace=True)
            
            st.success(f"✅ Successfully handled missing values in '{selected_column}'!")
            
            # Show remaining missing values
            remaining_missing = df[selected_column].isnull().sum()
            if remaining_missing > 0:
                st.warning(f"ℹ️ {remaining_missing} missing values remain in this column")
            else:
                st.success("🎉 No missing values remain in this column!")
                
        except Exception as e:
            st.error(f"Error applying changes: {str(e)}")
    
    return df

  
def show_unique_values(df):
    st.header("Unique Values Analysis")
    
    for column in df.columns:
        unique_count = df[column].nunique()
        unique_values = df[column].unique()
        
        with st.expander(f"{column}: {unique_count} unique values"):
        
            if unique_count < 50:  # Only show if not too many unique values
                col1,col2 = st.columns(2)
                with col1:
                    st.write("Value counts:")
                    st.dataframe(df[column].value_counts())   
            else:
                st.write(f"Too many unique values to display ({unique_count})")
            
def handle_duplicates(df):
    """Function to detect and handle duplicate rows in a DataFrame"""
    st.header("Duplicate Rows Analysis")
    
    # Count duplicates
    duplicates = df.duplicated().sum()
    
    if duplicates:
        st.warning(f"Found {duplicates} Duplicate Rows From {len(df)} Rows")
        duplicate_rows = df[df.duplicated()].sort_index()
        st.write("Preview of duplicate rows:")
        st.dataframe(duplicate_rows)
        
        if st.button("Remove Duplicates", type="primary"):
            df_cleaned = df.drop_duplicates(keep='first')
            st.success(f"Removed {duplicates} duplicate rows! Remaining: {len(df_cleaned)} Rows")
            st.write("Preview of cleaned data:")
            st.dataframe(df_cleaned)

            return df_cleaned
        
    else:
        st.success('✨ No Duplicate Rows Found')
        return df
    



def handle_outliers(df):
    st.header("Outliers Detection and Handling")
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        st.warning("No numeric columns found for outlier detection.")
        return df
    
    # Find columns with outliers
    columns_with_outliers = []
    outliers_info = {}
    
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_count = len(outliers)
        
        if outliers_count > 0:
            columns_with_outliers.append(column)
            outliers_info[column] = {
                'count': outliers_count,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
    
    if not columns_with_outliers:
        st.success("✨ No outliers found in any numeric columns!")
        return df
        
    # Display summary of columns with outliers
    st.subheader("Columns with Outliers:")
    for column in columns_with_outliers:
        st.write(f"- {column}: {outliers_info[column]['count']} outliers")
    
    # Use session state to track handled outliers
    if 'handled_outliers' not in st.session_state:
        st.session_state.handled_outliers = {}
    
    selected_column = st.selectbox(
        "Select a column to handle outliers", 
        columns_with_outliers
    )
    
    visualize_outliers(df, selected_column, lower_bound, upper_bound)
    # Get the pre-calculated bounds for the selected column
    lower_bound = outliers_info[selected_column]['bounds']['lower']
    upper_bound = outliers_info[selected_column]['bounds']['upper']

    st.write(f"**Recommended Bounds:**")
    st.write(f"Lower Bound: {lower_bound:.2f}")
    st.write(f"Upper Bound: {upper_bound:.2f}")
    
    # Use the stored bounds if column was previously handled
    if selected_column in st.session_state.handled_outliers and st.session_state.handled_outliers[selected_column]['handled']:
        stored_bounds = st.session_state.handled_outliers[selected_column]['bounds']
        lower_bound = st.number_input("Set Lower Bound", value=stored_bounds['lower'])
        upper_bound = st.number_input("Set Upper Bound", value=stored_bounds['upper'])
    # else:
    #     lower_bound = st.number_input("Set Lower Bound", value=lower_bound)
    #     upper_bound = st.number_input("Set Upper Bound", value=upper_bound)
   
    # Only show outliers if they haven't been handled yet
    if selected_column not in st.session_state.handled_outliers or not st.session_state.handled_outliers[selected_column]['handled']:
        outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
        
        st.write("Preview of outliers:")
        st.dataframe(outliers)
        
        # Choose handling method
        st.subheader("Select a method to handle outliers:")
        method = st.radio("Outlier Handling Method", options=['clip', 'drop'])
        
        # Apply the chosen method
        if st.button("Apply Changes"):
            df = handle_outliers_method(df, selected_column, lower_bound, upper_bound, method)
            # Update session state
            st.session_state.handled_outliers[selected_column] = {
                'handled': True,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            st.success("Outlier handling complete!")
            # Check if any outliers remain
            remaining_outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
            if len(remaining_outliers) > 0:
                st.warning(f"{len(remaining_outliers)} outliers remain in {selected_column}")
            else:
                st.success(f"All outliers in {selected_column} have been handled!")
    else:
        st.success(f"✅ Outliers in '{selected_column}' have already been handled")
        if st.button("Reset outlier handling for this column"):
            st.session_state.handled_outliers[selected_column]['handled'] = False
            st.experimental_rerun()
    
    return df

def handle_outliers_method(df, column, lower_bound, upper_bound, method='clip'):
    """Handles outliers based on the selected method."""
    if method == 'clip':
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        st.success(f"Outliers in '{column}' have been clipped to the defined bounds.")
    elif method == 'drop':
        df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, inplace=True)
        st.success(f"Outliers in '{column}' have been removed.")
    else:
        st.error("Invalid method for handling outliers.")
    return df


def handle_Categorical(df):
    df_copy = df.copy()
    """Handle categorical columns in the dataset."""
    st.header("🏷️ Handle Categorical Columns")
    Categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    if len(Categorical_columns) == 0:
        st.warning("No categorical columns found in the dataset.")
        return df

    st.subheader("Categorical Columns")
    st.table(Categorical_columns.tolist())

    # Let the user choose a column
    selected_column = st.selectbox("Select a categorical column to handle", Categorical_columns)

    # Provide options for handling the column
    method = st.radio(
        "Choose a method to handle the column:",
        options=["One-Hot Encoding", "Label Encoding", "See original"],
        index=0,
    )

    # Process the selected column based on the chosen method
    if method == "One-Hot Encoding":
        df_copy = pd.get_dummies(df, columns=[selected_column])
    elif method == "Label Encoding":
        label_encoder = LabelEncoder()
        df_copy[selected_column] = label_encoder.fit_transform(df[selected_column])
    else :
        pass
    
    st.write(df_copy.head())
    if st.button("Apply Changes"):
        df =  df_copy 
        if method == "One-Hot Encoding": 
            st.success(f"One-Hot Encoding applied to column: {selected_column}")
        elif method == "Label Encoding":
            st.success(f"Label Encoding applied to column: {selected_column}")
        else :
            pass
        
    return df    


def get_column_details(df):
    column_details = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        non_null_count = df[column].count()
        null_count = df[column].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        unique_values = df[column].nunique()
        sample_values = ", ".join(map(str, df[column].dropna().unique()[:5]))
        
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val = df[column].min()
            max_val = df[column].max()
            mean_val = df[column].mean()
            stats = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}"
        else:
            stats = "N/A"
        
        column_details.append({
            "Column Name": column,
            "Data Type": dtype,
            "Non-Null Count": f"{non_null_count} ({(non_null_count/len(df))*100:.1f}%)",
            "Null Count": f"{null_count} ({null_percentage:.1f}%)",
            "Unique Values": unique_values,
            "Sample Values": sample_values,
            "Statistics": stats
        })
    return pd.DataFrame(column_details)



ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



if uploaded_file is not None:
    try:  

        encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
        
        for encoding in encodings_to_try:
            try:
                csv_file = StringIO(uploaded_file.getvalue().decode(encoding))
                df= pd.read_csv(csv_file)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"Error reading CSV with {encoding} encoding: {str(e)}")
                continue  
    
        # Preview the dataset
        with st.expander("📊 Preview Dataset", expanded=False):
            st.dataframe(df.head())    
        
        # Show column details
        with st.expander("Columns Details" , expanded=False):
            st.dataframe(get_column_details(df))
                
        # Sidebar with file details
        with st.sidebar:
            st.success("File uploaded successfully!")
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("File Details:")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")

        if st.checkbox("Handle Duplicates"):
            df = handle_duplicates(df)

        if st.checkbox("Handle Missing Values"):
            df = handle_missing_values(df)

        if st.checkbox("Show Unique Values"):
            show_unique_values(df)
            
        if st.checkbox("Handle Outliers"):
            df = handle_outliers(df)  
            
        if st.checkbox("Handle Categorical Columns"):
            df = handle_Categorical(df)  
            
        if st.checkbox("Predict Spam"):
            st.title("Email Spam Classifier")
            input_sms = st.text_area("Enter the message")
            if st.button('Predict'):
                # 1. preprocess
                transformed_sms = transform_text(input_sms)
                # 2. vectorize
                vector_input = tfidf.transform([transformed_sms])
                # 3. predict
                result = model.predict(vector_input)[0]
                # 4. Display
                if result == 1:
                    st.header("Spam")
                else:
                    st.header("Not Spam (Ham)")
        # Allow users to download the modified data
        st.sidebar.header("Download Your Dataset")
        csv = df.to_csv(index=False)
        file_name = f"{uploaded_file.name.split('.')[0]}_modified.csv"
        st.sidebar.download_button(
            label="Download CSV 📥", 
            data=csv, 
            file_name=file_name, 
            mime="text/csv", 
            use_container_width=True 
        )
    except Exception as e:
        
      st.error(f"An error occurred: {e}")
      st.write(traceback.format_exc())
else:
    st.info("Please upload your dataset to begin.")


