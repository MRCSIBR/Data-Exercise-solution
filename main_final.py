import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import numpy as np

# Autor: Marcos D. Ibarra 
# Github: https://github.com/MRCSIBR

# Sidebar select box for choosing the page
page = st.sidebar.selectbox("Select Page", ["Data Exercise ETL", "News Classifier"])
st.sidebar.subheader("Autor: Marcos D. Ibarra")

# Define the main function for the Data Exercise ETL page
def data_etl_page():
    
    st.title("Data Exercise 1: ETL Geopositional")
    # Create a file uploader widget to load the CSV file
    uploaded_file = st.file_uploader("Upload CSV with geo data", type="csv")

    if uploaded_file is not None:
        
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Limpiar el dataset de 'guiones' para cargar correctamente
        data_subset = data[data['geo_points'].notnull() & (data['geo_points'] != '-')]
        st.write(data_subset.head(10))
        
        # Convert the 'geo_points' column to Point objects
        point_objects = [Point(float(x.split(' ')[1][1:]), float(x.split(' ')[2][:-1])) for x in data_subset['geo_points']]
        
        gdf_points = gpd.GeoDataFrame(geometry=point_objects)
        
        
        # Calculate the center latitude and longitude based on the concentration of datapoints
        center_lat = np.median(gdf_points.geometry.y)
        center_lon = np.median(gdf_points.geometry.x)
        
        # Plot the GeoDataFrame using plotly express
        fig = px.scatter_mapbox(gdf_points, lat=gdf_points.geometry.y, lon=gdf_points.geometry.x,
                                center=dict(lat=center_lat, lon=center_lon),
                                zoom=3, height=600)

        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

# Define the main function for the News Classifier page
def news_classifier_page():
    
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report

    # Function to load data
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    # Function to train and test the model
    @st.cache_data
    def train_and_test_model(data):
        X_train, X_test, y_train, y_test = train_test_split(
            data['content'], 
            data['type'], 
            test_size=0.2, 
            random_state=42
        )
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        return report_df

    # Function to plot bar graph
    def plot_bar_graph(data):
        fig, ax = plt.subplots()
        ax.bar(data.index, data['f1-score'])
        ax.set_xlabel('Class')
        ax.set_ylabel('F1-Score')
        ax.set_title('Classification Report')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Streamlit widgets
    st.title('Data Exercise 2: News classifier')
    st.write('Upload a CSV file for news classification. Then press `Classify News` button.')

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if st.button('Classify News'):
            st.write('First 10 rows of the dataset:')
            st.write(data.head(10))
            report_df = train_and_test_model(data)
            st.table(report_df)
            plot_bar_graph(report_df)

# Check the selected page and display the corresponding content
if page == "Data Exercise ETL":
    data_etl_page()

elif page == "News Classifier":
    news_classifier_page()
