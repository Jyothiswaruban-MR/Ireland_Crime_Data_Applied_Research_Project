Ireland Crime Data – Applied Research Project

Live Application URL
https://irishcrimedatatest.streamlit.app/

This project is an interactive crime analytics dashboard for Ireland, built using Python and Streamlit.
It was developed as part of an Applied Research Project to explore long-term crime trends using publicly available data from the Central Statistics Office (CSO), Ireland.

The main goal of the project is to make Irish crime data easy to explore, understand, and interpret through interactive visualisations, filters, and simple narrative insights.

Rather than presenting static charts, this dashboard allows users to ask questions of the data — such as how crime has changed over time, how it varies by region or Garda division, and which offence types are most common.

**What the dashboard allows you to do**

Using the dashboard, you can:

1.Explore crime trends over time at a national level
2.Compare crime across regions and Garda divisions
3.Analyse crime by offence type
4.View Garda station-level patterns
5.Apply global filters for year range, region, division, and offence
6.Read automatically generated summary insights
7.View forecasted crime results for recent years (where available)

**Data source**

All data used in this project comes from the Central Statistics Office (CSO), Ireland.

The original CSO datasets were:

1.Cleaned
2.Standardised
3.Enriched with regional and divisional information
4.Aggregated into analysis-ready formats

To keep the repository lightweight and reproducible, raw CSO datasets are not included.
Only the final cleaned and aggregated datasets required to run the dashboard are stored in this repository.

How the data is processed

At a high level, the data workflow is:

1.Ingest raw CSO crime datasets
2.Clean and standardise key fields (years, regions, offence labels)
3.Enrich records with Garda region and division information
4.Aggregate data by year, region, division, offence, and station
5.Generate final analytical CSV files
6.Visualise and analyse the data using Streamlit

This separation between raw data, processing scripts, and final datasets helps ensure clarity, reproducibility, and transparency.

**PROJECT STRUCTURE**

├── app.py                    # Main Streamlit application
├── data/
│   └── cleaned/              # Final analytical datasets used by the app
├── data_cleaning.py          # Data cleaning logic
├── data_merging.py           # Dataset merging and enrichment
├── data_aggregation.py       # Aggregation scripts
├── crime_forecast.py         # Forecast generation logic
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore

**Technologies used**
-Python
-Streamlit
-Pandas
-Altair
-Pathlib
-Scikit-learn (used offline for model training and experimentation)

Running the project locally
If you would like to run the dashboard locally:

pip install -r requirements.txt
streamlit run app.py


**Limitations**
-Forecasts are illustrative and based on historical patterns
-The dashboard is intended for analysis and exploration, not as an official crime prediction tool

**Future improvements**
-Possible future enhancements include:
-More advanced forecasting techniques
-Geographical maps for spatial analysis
-Downloadable filtered datasets
-Additional performance and model evaluation metrics
-Expanded narrative explanations within the app

Author
Jyothiswaruban Madurai Ravishankar
MSc – Information System with Computing 
Dublin Business School

