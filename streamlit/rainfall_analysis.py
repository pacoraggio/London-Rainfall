import streamlit as st
import pandas as pd
import numpy as np


data = pd.read_pickle('./data/london_tp_daily_data.pkl')

data = data[['date', 'year', 'month', 'day', 'tp_mm_daily_sum', 'month_int']]
years = np.sort(data['year'].unique())
select_years = set(years)

option = st.selectbox(
    "Select Year",
    select_years,
)



st.write(f"you selected {option} year")
# st.write("Here's our first attempt at using data to create a table:")
# st.write(data.dtypes)