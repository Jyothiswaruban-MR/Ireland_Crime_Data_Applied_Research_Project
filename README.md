Ireland Crime Data – Applied Research Project is an interactive crime analytics dashboard developed using Python and Streamlit as part of an MSc Applied Research Project at Dublin Business School (DBS) within the Computing and Information Systems discipline. The project investigates long-term trends in recorded crime in Ireland using publicly available, anonymised data from the Central Statistics Office (CSO), with the primary aim of improving accessibility, interpretability, and exploratory analysis of crime statistics for non-technical and policy-oriented users. Rather than presenting static charts, the dashboard enables interactive exploration of crime data, including analysis of national trends over time, comparison across Garda regions and divisions, examination of offence categories and Garda station-level patterns, application of global filters for year range, region, division, and offence type, and the presentation of automatically generated narrative summaries to support interpretation. To ensure reproducibility while keeping the repository lightweight, raw CSO datasets are retrieved programmatically via the CSO Open Data API and are not stored within the repository, with only the final cleaned and aggregated analytical datasets retained locally for dashboard execution. The data processing workflow consists of programmatic data ingestion, cleaning and standardisation of key fields such as years, offence labels, and regional identifiers, enrichment with Garda region and division metadata, aggregation by year, region, division, offence type, and Garda station, and optional generation of illustrative short-term crime forecasts based solely on historical trends. The analytical pipeline is executed by first running the data ingestion module, followed sequentially by data cleaning, merging, and aggregation scripts, with forecasting logic optionally generated through a dedicated forecasting module prior to launching the interactive application using Streamlit. Forecast outputs are explicitly framed as exploratory and illustrative rather than predictive or operational, and the dashboard is intended to support analysis and understanding rather than official crime forecasting or decision-making. The project adheres to DBS Applied Research guidelines, demonstrates clear integration between research objectives and applied artefact development, and addresses ethical considerations associated with public-sector crime data through aggregation, transparency, and acknowledgment of data and modelling limitations, with potential future enhancements including advanced forecasting techniques, spatial mapping, downloadable filtered datasets, and expanded narrative explanations within the application.

->Download the raw CSO crime dataset (programmatic)

python
from datasets import download_recorded_crime
download_recorded_crime(force=True)

->Run the data processing pipeline

python data_cleaning.py
python data_merging.py
python data_aggregation.py

->Generate the Crime forecasts

python crime_forecast.py

->Launch Streamlit dashboard

streamlit run app.py
