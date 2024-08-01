import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import joblib
import time
import io
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.impute import SimpleImputer
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load the custom CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the CSS
load_css('styles.css')
 
def plot_crosstab_heatmap(df, feature1, feature2):
    # Create crosstab
    crosstab = pd.crosstab(df[feature1], df[feature2])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(crosstab, cmap='Blues')

    # Add color bar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(crosstab.columns)))
    ax.set_yticks(np.arange(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticklabels(crosstab.index)

    plt.xlabel(feature2)
    plt.ylabel(feature1)
    plt.title(f'Crosstab Heatmap of {feature1} vs {feature2}')

    # Add text annotations
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            ax.text(j, i, crosstab.iloc[i, j], ha='center', va='center', color='black')

    return fig


def report_to_dataframe(report_dict):
    return pd.DataFrame(report_dict).transpose()

def plot_classification_report(report_df):
    fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size
    ax.axis('off')  # Hide axes

    # Create a table and add it to the figure
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale table

    plt.title('Classification Report')
    return fig

def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    return plt

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    return plt


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')

    return fig

def plot_class_distribution(y):
    plt.figure()
    class_counts = y.value_counts()
    plt.bar(class_counts.index, class_counts.values, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    return plt

def plot_feature_importance(coefficients, feature_names):
    plt.figure()
    plt.barh(feature_names, coefficients, color='salmon')
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance')
    return plt

def load_model(uploaded_file):
    try:
        model = joblib.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def plot_summary_statistics(df):
    """
    Plots summary statistics (mean and standard deviation) for each feature in the DataFrame.

    Parameters:
    - df: pd.DataFrame - The dataset for which to plot the statistics.
    """
    summary = df.describe()
    
    # Extract mean and standard deviation
    mean = summary.loc['mean']
    std = summary.loc['std']

    plt.figure(figsize=(14, 7))

    # Plot mean
    plt.bar(mean.index, mean, color='skyblue', alpha=0.6, label='Mean')
    
    # Plot standard deviation
    plt.errorbar(mean.index, mean, yerr=std, fmt='o', color='red', capsize=5, label='Standard Deviation')

    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Summary Statistics: Mean and Standard Deviation')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_correlation_matrix(df):
    """
    Plots a correlation matrix heatmap for the features in the DataFrame.

    Parameters:
    - df: pd.DataFrame - The dataset for which to plot the correlation matrix.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(20, 16))
    
    # Create a heatmap
    cax = plt.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add color bar
    plt.colorbar(cax)
    
    # Set labels
    plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90,fontsize=7)
    plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns,fontsize=7)
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


def preprocessed_df(df,target):
    # Universal Preprocessing

    # 1. Missing Values
    for i in df.columns:
        if i!=target:
            percent_missing = df[i].isna().mean() * 100

        # Convert column to 2D array
        data = df[i].values.reshape(-1, 1)

        if percent_missing > 20:
            df = df.drop(i, axis=1)
        else:
            if df[i].dtype == 'object':
                imputer = SimpleImputer(strategy='most_frequent')
                df[i] = imputer.fit_transform(data).ravel()
            else:
                if percent_missing < 5:
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='median')
                df[i] = imputer.fit_transform(data).ravel()

    # 2. Convert to numerical format if still object type
    for i in df.columns:
        if i!=target:
            if df[i].dtype == 'object':
                label_encoder = LabelEncoder()
                df[i] = label_encoder.fit_transform(df[i])

        

    df_cleaned = df.dropna(subset=[target])


    return df


def load_hyperparam(model):
    # Hyperparameters for Logistic Regression
    global hyperp,key_dict,test_size

    st.sidebar.title("Model Hyperparameters")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

    hyperp = model.get_params()
    # hyperp['max_iter'] = 1000 

    key_dict={}
    for key,values in hyperp.items():
        check=False
        if isinstance(values,str):

            if values.isdigit():
                # values = int(values)
                check=True
        save_key=key
        # st.text(key)
        values = st.sidebar.text_input(f"Enter the {key} value",f"{values}")
        key_dict[key]=values
        if values=="None":
            key_dict[key]=None
        if values=="False":
            key_dict[key]=False
        if values=="True":
            key_dict[key]=True
        
        # st.text(key)
        
        try:
            float(values)
            # st.text("Float")
            key_dict[key]= float(values)
        except:
            pass
    return key_dict,test_size
    

def make_predictions(df,key_dict,test_size,target,model):

    st.write(f"Model type: {type(model)}") 
    X= df.drop(target,axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model is None:
        # Initialize the Linear Regression model
        model = LogisticRegression(C=key_dict["C"],
                               class_weight=key_dict["class_weight"],
                               dual=key_dict["dual"],
                               fit_intercept=key_dict["fit_intercept"],
                               penalty=key_dict["penalty"],
                               intercept_scaling=key_dict["intercept_scaling"],
                               max_iter=int(key_dict["max_iter"]),
                               multi_class=key_dict["multi_class"],
                               n_jobs=key_dict["n_jobs"],
                               random_state=key_dict["random_state"],
                               solver=key_dict["solver"],
                               tol=key_dict["tol"],
                               verbose=int(key_dict["verbose"]),
                               warm_start=key_dict["warm_start"])

    # Train the model
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    col1, col2 = st.columns(2)

    with col1:
        # Display the results
        st.write("### Accuracy:", accuracy_score(y_test,y_preds))
        st.write('### Classification Report: ')
        class_report=classification_report(y_test,y_preds)
        # st.code(class_report,language="text")

        # Convert classification report to DataFrame
        report_df = report_to_dataframe(classification_report(y_test, y_preds, output_dict=True))

        # Plot and display the classification report
        fig = plot_classification_report(report_df)
        st.pyplot(fig)
        
        # Plotting Precision-Recall curve
        st.write("### Precision-Recall curve")
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        pr_fig = plot_precision_recall_curve(y_test, y_probs)
        st.pyplot(pr_fig)

        # Plotting Class Distribution
        st.write("### Class Distribution")
        dist_fig = plot_class_distribution(y)
        st.pyplot(dist_fig)

        # plot_crosstab_heatmap(df,'Diagnosis','BehavioralProblems')
    
    with col2:
        st.write("### Confusion Matrix")
        con_mat = confusion_matrix(y_test,y_preds)
        # st.code(con_mat,language="text")

        class_names = list(map(str, np.unique(y)))

        # Plotting Confusion Matrix
        fig = plot_confusion_matrix(con_mat, class_names)
        st.pyplot(fig)
    
        # Plotting ROC curve
        st.write("### ROC Curve")
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        roc_fig = plot_roc_curve(y_test, y_probs)
        st.pyplot(roc_fig)
    
        # Plotting Feature Importance
        st.write("### Feature Importance")
        coefficients = model.coef_[0]  # Coefficients from the trained model
        feature_names = X.columns
        feat_fig = plot_feature_importance(coefficients, feature_names)
        st.pyplot(feat_fig)
    # plot_crosstab_heatmap(alzheimer_data,'Diagnosis','BehavioralProblems')



def plot_basic_histogram(data, feature, bins=30):
    """
    Plots a basic histogram for a single feature to show its distribution.

    Parameters:
    - data: list or pd.Series - The data for the feature.
    - feature: str - The label for the feature.
    - bins: int - Number of bins for the histogram.
    """
    # Ensure data is a numerical type
    if not pd.api.types.is_numeric_dtype(data):
        data = pd.to_numeric(data, errors='coerce')  # Convert to numeric, setting errors to NaN
    
    # Drop NaN values if any
    data = data.dropna()

    if data.empty:
        st.error(f"No numerical data to plot for {feature}.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')

    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    st.pyplot(plt)
    plt.close()


# "-----------------------------------------------------------------------------------------------------------"
# "-----------------------------------------------------------------------------------------------------------"
# "-----------------------------------------------------------------------------------------------------------"
# "-----------------------------------------------------------------------------------------------------------"
# "-----------------------------------------------------------------------------------------------------------"
# "-----------------------------------------------------------------------------------------------------------"


def main():

    # Initialize session state variables if not already set
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None


    st.title("Model Visualization Dashboard")

    # Sidebar Code
    st.sidebar.subheader("Upload a CSV file") 
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file",type="csv")
    target = st.sidebar.text_input('Enter the Target Column')
    if uploaded_file is not None:
        if target!="":
            if uploaded_file.type == 'text/csv':
                df = pd.read_csv(uploaded_file)
                # last= df.columns[-1]
                # df=df.drop(last,axis=1)
                st.success("CSV file loaded successfully!")
            else:
                st.write("Error")
                st.error("The file type is not .csv")
        else:
            st.error("Enter the target variable")
    else:
        st.error("Please enter a CSV file")
    

    tab_1,tab_2 = st.tabs(["Input Data", "Predicted Data"])
    
    if uploaded_file is not None and target!="":
        # Input Data Visualization
        with tab_1:
            st.header("This is Input Feature Visualization")
            dist,describedf,corr = st.tabs(['Distributon','Data Frame Description','Correlation Matrix'])
            with dist:
                for i in df.columns:
                    
                    plot_basic_histogram(df[i],i,bins=30)
            with describedf:
                plot_summary_statistics(df)         

            with corr:
                numeric_df = df.select_dtypes(include=[np.number])
                plot_correlation_matrix(numeric_df)
        
        # Use session_state to manage state between reruns
        if 'trained' not in st.session_state:
            st.session_state.trained = False
        # Sidebar Buttons

        sidebut1,sidebut2 = st.sidebar.columns(2)

        with sidebut1:
            if st.button('Train a model'):
                st.sidebar.write("Training model...")
                # Importing preprocessed data set

                df_copy=df
                clean_df = preprocessed_df(df,target)

                # # Fit model
                model = LogisticRegression()
                key_dict,test_size = load_hyperparam(model)  
                

                # model.fit(X_train,y_train) 
                # st.write(model.score(X_test,y_test))     

                # Predicted Data Visualization
                with tab_2:
                    st.header("Data Predictions and Metrics")
                    make_predictions(clean_df,key_dict,test_size,target,model=None)   
                    st.session_state.trained = True


        with sidebut2:
            st.sidebar.write('----------------------------------')
            uploaded_model = st.sidebar.file_uploader("Upload a model", type="pkl")
            if st.button('Upload a model'):
                
                
                # st.write(model.get_params())
                if uploaded_model is not None:
                    # st.write("DPNEEE")
                    # st.sidebar.write("dONee")
                    model = load_model(uploaded_model)
                    if model is not None:
                        # st.write("DPNEEE")
                        # st.sidebar.write("dONee")
                        st.success("Model loaded successfully!")
                    
                        # Importing preprocessed data set
                        df_copy = df
                        clean_df = preprocessed_df(df, target)

                        # Load hyperparameters
                        st.sidebar.success('Loading hyperp')
                        key_dict, test_size = load_hyperparam(model)
                        st.success(key_dict)
                        # Predicted Data Visualization
                        with tab_2:
                            st.success('Helo')
                            st.header("Data Predictions and Metrics")
                            st.write('Predictions below:')
                            make_predictions(clean_df, key_dict, test_size, target, model=model)
                            st.session_state.trained = True

                    else:
                        st.error('model = load_model(uploaded_model) not working')

                    
                else:
                    st.error("Failed to load the model. Please check the file and try again.")
                

                    
        #Button to save the trained model
        # sidebut_save = st.sidebar.button('Save the Model')
        # if sidebut_save:
        if st.session_state.trained == True:
            save_path = st.sidebar.text_input('Enter the filename to save the model')
        

        if st.sidebar.button('Save the model'):
                if st.session_state.trained:
                    # Save model to a file
                    model = st.session_state.model



                    # Save model to a BytesIO object
                    buffer = io.BytesIO()
                    joblib.dump(model, buffer)
                    buffer.seek(0)  # Rewind the buffer to the beginning


                    

                    if save_path is not None:

                        # Provide a download button
                        st.download_button(
                        label="Download Model",
                        data=buffer,
                        file_name=save_path,
                        mime='application/octet-stream'
                        )
                    else:
                        st.error("Enter a file name")


                    # if save_path is not None and save_path!='trained_model.pkl' :
                    #     with open(save_path, 'wb') as file:
                    #         joblib.dump(model, file)
                    #     st.success(f"Model saved as {save_path}")

                    

                    
                else:
                    st.warning("No model has been trained yet.")
            


        # Checking for regression/ classification
        if(df[target].value_counts().size < len(df)/2):
            pass
            # st.success('Classification Problem Detected')
        else:
            pass
            # st.success('Regression Problem Detected')

        # Creating tabs for the Input Data and Predicted Data
        
        if df.shape[1] == 0:
            st.warning("No features available. Please upload a valid CSV file.")
        elif st.session_state.trained:
            st.write("Model has been trained. Use the 'Train a model' button to retrain with new parameters.")
        else:
            st.write("Upload a model or train a new model.")


        
        
            
        


if __name__ == "__main__":
    main()
