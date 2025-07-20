import streamlit as st
import pandas as pd
import numpy as np


from src.data_management import load_data, process_data_for_year, load_parquet_data
from src.data_visualization import plot_monthly_aggregate_comparison, plot_monthly_aggregate_stacked, plot_monthly_aggregate_overlaid
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(
    page_title="London and Apulia Monthly Precipitation Analysis",
    page_icon="🌧️",
    layout="wide",  # This sets wide mode
    initial_sidebar_state="expanded"
)


st.title("London vs Apulia -- Monthly Precipitation Analysis")

london_monthly_aggregate = load_parquet_data('./data/london_monthly_aggregate.parquet')
apulia_monthly_aggregate = load_parquet_data('./data/apulia_monthly_aggregate.parquet')

# st.write(london_monthly_aggregate.head())
# st.write(apulia_monthly_aggregate.head())


col1, col2 = st.columns([1,2])


with col1:
    metric = st.radio(
        'Choose metric',
        ['Average', 'Median', 'Sum'],
        captions=[
            'Precipitation Mean',
            'Precipitation Median',
            'Total Precipitation'
        ],
        horizontal=True
    )

with col2:
    visualization_type = st.radio(
        "Choose Visualization Type",
        ['option 1', 'option 2', 'option 3'],
        captions=['One on top the other', 
                'Side by side',
                'I haven\'t the slightest idea'],
                horizontal=True
    )
st.write('---')

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Monthly Precipitation Comparison")
    if visualization_type == 'option 1':
        fig = plot_monthly_aggregate_comparison(london_monthly_aggregate, apulia_monthly_aggregate,
                                        label1='London',
                                        label2='Apulia',
                                        feature_label=metric,
                                        bar_width=0.35,
                                        location='Comparison',
                                        fig_width=900,
                                        fig_height=600,
                                        dark_mode=False)
        st.plotly_chart(fig, use_container_width=True)
    elif visualization_type == 'option 2':
        fig = plot_monthly_aggregate_stacked(london_monthly_aggregate, apulia_monthly_aggregate,
                                    label1='London',
                                    label2='Apulia',
                                    feature_label=metric,
                                    bar_width=0.35,
                                    location='Comparison',
                                    fig_width=900,
                                    fig_height=600,
                                    dark_mode=True)
        st.plotly_chart(fig, use_container_width=True)
    elif visualization_type == 'option 3':
        fig = plot_monthly_aggregate_overlaid(london_monthly_aggregate, apulia_monthly_aggregate,
                                    label1='London',
                                    label2='Apulia',
                                    feature_label=metric,
                                    bar_width=0.35,
                                    location='Comparison',
                                    fig_width=900,
                                    fig_height=600,
                                    dark_mode=True)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    metric_column = metric
    st.subheader(f"Monthly {metric_column}")
    st.markdown("London (LND) vs Apulia (APL) Summary Table")
    metric_dict = {
        'Sum': 'month_sum',
        'Average': 'month_avg',
        'Median': 'month_median'
    }

    cols = ['month', metric_dict[metric], 'month_min', 'month_max']

    # First try
    # df_display = (london_monthly_aggregate[cols].merge(apulia_monthly_aggregate[cols], on='month', suffixes=('_london', '_apulia'))
    #               .rename(columns={
    #                   'month': 'Month',
    #                   f'{metric_dict[metric]}_london': f'London {metric_column}',
    #                   f'{metric_dict[metric]}_apulia': f'Apulia {metric_column}',
    #                   'month_min_london': 'London Min',
    #                   'month_max_london': 'London Max',
    #                   'month_min_apulia': 'Apulia Min',
    #                   'month_max_apulia': 'Apulia Max'
    #               }))[['Month',f'London {metric_column}', f'Apulia {metric_column}',  
    #                    'London Min', 'Apulia Min',
    #                    'London Max',  'Apulia Max']]
    


    # # st.write(df_display.columns)
    # # Create a dictionary of custom column headers with line breaks
    # custom_headers = {
    #     f'London {metric_column}': f'London<br>{metric_column}',
    #     f'Apulia {metric_column}': f'Apulia<br>{metric_column}',
    #       'London Min': 'London <br> Min',
    #       'Apulia Min': 'Apulia<br>Min',
    #       'London Max': 'London<br>Max',
    #       'Apulia Max': 'Apulia<br>Max',
    #       'Month': 'Month'
    #       }
    
    # # st.write(london_monthly_aggregate[cols])
    # gb = GridOptionsBuilder.from_dataframe(df_display.reset_index(drop=True))
    # gb.configure_default_column(filterable = False, sortable = True, resizable = True)

    # for col in df_display.columns:
    #     if col in custom_headers:
    #         gb.configure_column(col, header_name=custom_headers.get(col, col), wrapText=True, autoHeight=True)
    #     else:
    #         gb.configure_column(col, wrapText=True, autoHeight=True)
    # grid_options = gb.build()

    # AgGrid(df_display, 
    #        gridOptions=grid_options, 
    #        theme='streamlit',
    #        fit_columns_on_grid_load=True,
    #         allow_unsafe_jscode=True )   
    #st.dataframe(df_display.reset_index(drop=True), use_container_width=False)
    # st.dataframe(apulia_monthly_aggregate, use_container_width=True)

    # Second try
    # Custom CSS for centered headers
    centered_header_style = """
    <style>
    .ag-header-cell-label {
        justify-content: center !important;
        text-align: center !important;
        white-space: normal !important;  /* wrap header text */
    }
    </style>
    """

      # Custom headers with line breaks
    abbreviation_dict = {
        'Average': 'Avg',
        'Sum': 'Sum',
        'Median': 'Med'
    }

    df_display = (
    london_monthly_aggregate[cols]
    .merge(apulia_monthly_aggregate[cols], on='month', suffixes=('_london', '_apulia'))
    .rename(columns={
        'month': 'Month',
        f'{metric_dict[metric]}_london': f'LND | {abbreviation_dict[metric_column]}',
        f'{metric_dict[metric]}_apulia': f'APL | {abbreviation_dict[metric_column]}',
        'month_min_london': 'LND | Min',
        'month_max_london': 'LND | Max',
        'month_min_apulia': 'APL | Min',
        'month_max_apulia': 'APL | Max'
    })
    )[['Month', f'LND | {abbreviation_dict[metric_column]}', f'APL | {abbreviation_dict[metric_column]}',
       'LND | Min', 'APL | Min', 'LND | Max', 'APL | Max']]

    london_color = "#0072B2"
    apulia_color = "#E69F00"

    highlight_min_style = {
    'backgroundColor': '#d4f4dd',  # light green
    'fontWeight': 'bold'
    }

    highlight_max_style = {
    'backgroundColor': '#f9d4d4',  # light red
    'fontWeight': 'bold'
    }

    custom_headers = {
        f'London {abbreviation_dict[metric_column]}': f'LND | {abbreviation_dict[metric_column]}',
        f'Apulia tt {metric_column}': f'APL | {metric_column}',
        'London Min': 'LND | Min',
        'Apulia Min': 'APL | Min',
        'London Max': 'LND | Max',
        'Apulia Max': 'APL | Max',
        'Month': 'Month'
    }
        # GridOptionsBuilder
    gb = GridOptionsBuilder.from_dataframe(df_display.reset_index(drop=True))
    gb.configure_default_column(
        filterable=False,
        sortable=True,
        resizable=True,
        headerClass='centered-header'
    )

    # Override each column to disable filtering
    for col in df_display.columns:
        style = {}

        if 'LND' in col:
            style = {
            'backgroundColor': london_color,
            'color': 'white',
            'fontWeight': 'bold'
            }
        elif 'APL' in col:
            style = {
            'backgroundColor': apulia_color,
            'color': 'black',
            'fontWeight': 'bold'
            }

        gb.configure_column(
            col,
            header_name=custom_headers.get(col, col),
            filter=False,  # Explicitly disable filter here
            cellStyle=style 
        )

    grid_options = gb.build()

    st.markdown("""
    <style>
        .ag-header-cell-label {
            justify-content: center !important;
            text-align: center !important;
            white-space: normal !important;
            line-height: 1.2 !important;
        }

        .ag-header-cell-text {
            word-wrap: break-word !important;
            max-width: 100px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # AgGrid display with auto height and dynamic sizing
    AgGrid(
        df_display,
        gridOptions=grid_options,
        theme='streamlit',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        height=400
    )
