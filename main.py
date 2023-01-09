# Data Management
import pandas as pd, numpy as np, re
from pandas_profiling import ProfileReport as pp

# Visualizations
import seaborn as sns, matplotlib.pyplot as plt, plotly.express as px
import plotly.io as pio
pio.renderers.default = 'chrome'

# Settings & Other
from joblib import dump, load
import warnings

# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from common import set_page_container_style
st.set_page_config(layout="wide", page_icon="chart_with_upwards_trend", page_title="Big Data Visions - Dabbco Bid Demo")
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

# Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import shap

# Utility load function
def load_file(file_path):    
    return load(file_path)

st.title("""Big Data Visions - Dabbco Bid Demo""")

st.markdown("""
  <style>
    .css-renyox e1fqkh3o3 {
      margin-top: -75px;
    }

  </style>
""", unsafe_allow_html=True)

st.markdown("""
  <style>
    .css-18e3th9.egzxvld2 {
      margin-top: -75px;
    }

  </style>
""", unsafe_allow_html=True)

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)



# Load model, training data, enconded training data, prediction data and cache it
@st.cache
def fetch_encoded_data():
  df = load("encoded_training_df.joblib")
  return df

def fetch_train_data():
  df = load("training_df.joblib")
  return df

def fetch_model():
  model = load("xgb_model.joblib")
  return model

encoded_training_df = fetch_encoded_data()
training_df = fetch_train_data()
model = fetch_model()

# Load EDA report
path_to_html = "./report.html" 
with open(path_to_html,'r') as f: 
    report = f.read()

# # Display Training Data
# is_check = st.checkbox("Display Training Data")
# if is_check:
#     st.dataframe(training_df.head(5))

# st.set_option('deprecation.showPyplotGlobalUse', False)

tab1, tab2, tab3 = st.tabs(["How this Works", "What the Data Looks like", "Model Findings"])

with tab1:
  st.markdown("""
  ## How this Works
  In machine learning, classification refers to a predictive modeling problem where a class label is predicted for a given example of input data.

  Examples of classification problems include:

  - Given an email received, classify if it is spam or not.
  - Given a handwritten character, classify it as one of the known characters.
  - Given recent user behavior, classify as churn or not.

  From a modeling perspective, classification requires a training dataset with many examples of inputs and outputs from which to learn.

  A model will use the training dataset and will calculate how to best map examples of input data to specific class labels. 
  As such, the training dataset must be sufficiently representative of the problem.

  You can use the toolbar to the left to "input your bid", and see if it is predicted to win the award with a confidence level. You can also click through the 
  tabs to examine the data deeper. The third tab displays the model findings and what bid inputs are most important to win or lose an award. 

  """)

with tab2:
  st.subheader("Sample Data")
  # Display Training Data
  is_check = st.checkbox("Display sample data")
  if is_check:
      st.dataframe(training_df.head(5))

  st.set_option('deprecation.showPyplotGlobalUse', False)

  # # Create download function
  # @st.cache
  # def get_table_download_link_csv(df):
  #     csv = df.to_csv().encode()
  #     b64 = base64.b64encode(csv).decode()
  #     href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download CSV File</a>'
  #     return href

  #st.markdown(get_table_download_link_csv(training_df), unsafe_allow_html=True)

  st.subheader("Exploratory Data Analysis")
  is_check = st.checkbox("Display EDA report")
  if is_check:
    components.html(report, height=1200, width=1200, scrolling=True)

  st.subheader("Natural Language Processing with Bid Remarks")
  is_check = st.checkbox("Display NLP Visuals")
  if is_check:
    st.write("Remarks Word Cloud")
    st.image('dabbco_wordcloud.png', width=700)

    st.write("Remarks Common Word Counts")
    st.image('dabbco_topwords.png', width=700)
    st.image('dabbco_topbigrams.png', width=700)
    st.image('dabbco_toptrigrams.png', width=700)


with tab3:
  # Load SHAP Values
  shap_values = load('shap_values.joblib')

  st.subheader('Important Data Points with SHAP')
  st.markdown(
      """<a href="https://shap.readthedocs.io/en/latest/index.html">SHAP Explained</a>""", unsafe_allow_html=True,
  )

  values = st.slider("Number of SHAP Features", min_value=5, max_value=40, value=5)

  # the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
  #shap.plots.beeswarm(shap_values, max_display=values)
  #shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0))
  shap.plots.bar(shap_values, max_display=values)
  st.pyplot()

st.sidebar.image('Transparent-PNG.png', width=200)
st.sidebar.title("Input your bid")
st.sidebar.write("Enter a bid by filling out the below form and select 'Predict Your Bid' to see if your bid is predicted to win the award with an associated confidence percentage.")

# Keep only alphanumeric values, strip extra space, keep it lowerspace
def process_descriptions(txt):
    return re.sub("[^a-zA-Z] +", "", txt).lower().strip()   

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

# Create User Input
# Unique values from dataframe
gen_contract = training_df['General_Contractor'].sort_values(ascending=True).unique()
addendum = training_df['Addendum'].unique()
est = training_df['EST'].sort_values(ascending=True).unique()

# Select values
gen_contract_select = st.sidebar.selectbox('Select General Contractor', gen_contract)
remarks_select = st.sidebar.text_input('Remarks', value='Input Remarks')
addendum_select = st.sidebar.selectbox('Select Addendum', addendum)
est_select = st.sidebar.selectbox('Select Estimator', est)

if st.sidebar.button("Predict Your Bid!"):
    
  # Store Values to DF
  data = {'General_Contractor':[gen_contract_select], 
          'Remarks':[remarks_select], 
          'Addendum':[addendum_select], 
          'EST':[est_select], }
  features = pd.DataFrame(data)

  input_df = features.copy()

  st.subheader("Model")
  st.write('Your Bid')
  st.dataframe(input_df.head())

  # Create term counts and one hot encoding objects
  vec = CountVectorizer(decode_error='ignore', strip_accents='unicode', lowercase=True, stop_words='english')
  ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False)

  # Columns to use for modeling -- NOT stakeholder given
  model_cols = ['General_Contractor', 'Remarks', 'Addendum', 'EST']

  # Transform the data
  cat_data = pd.DataFrame(data=ohe.fit_transform(input_df[[x for x in model_cols if x not in ['Remarks']]]), columns=list(ohe.get_feature_names_out()))
  text_data = pd.DataFrame(data=vec.fit_transform(input_df['Remarks'].apply(process_descriptions)).todense(), columns=list(vec.get_feature_names_out()))
  text_data = text_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

  X_pred = pd.concat([cat_data, text_data], axis=1)

  # Determine which columns are missing between the training and the open po data
  missing_cols = set(encoded_training_df.columns) - set(X_pred.columns)

  # Create the missing columns in the open po data
  for c in missing_cols:
      X_pred[c] = 0

  # Ensure that the columns are in the correct order
  X_pred = X_pred[encoded_training_df.columns]

  # Make sure there are equal columns in everything
  assert(encoded_training_df.shape[1] == X_pred.shape[1])

  # Make predictions on the open po data (and grab the probabilities of the predictions)
  predictions = model.predict(X_pred)
  probabilities = model.predict_proba(X_pred)

  # Transpose the results of predict_proba
  x, y = probabilities.T

  # Create the dataframe
  temp_df = pd.DataFrame({'PREDICTION': predictions, 'Class 0': x, 'Class 1': y})

  # Determine the probability of the prediction
  temp_df['CONFIDENCE'] = temp_df[['Class 0', 'Class 1']].max(axis=1)
  temp_df['CONFIDENCE'] = str(round(float(temp_df['CONFIDENCE'][0])*100)) + '%'

  # Drop unnecessary columns
  temp_df.drop(['Class 0', 'Class 1'], inplace=True, axis=1)
  # Replace 1 and 0 with Late vs On Time
  temp_df['PREDICTION'].replace(1, 'Award', inplace=True)
  temp_df['PREDICTION'].replace(0, 'No Award', inplace=True)

  # Display Predictions
  st.write('Bid Prediction')
  st.dataframe(temp_df.head())
