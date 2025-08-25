import streamlit as st
import pandas as pd
import numpy as np


from src.data_management import load_data, process_data_for_year, load_parquet_data
from src.data_visualization import plot_monthly_aggregate_comparison, plot_monthly_aggregate_stacked, plot_monthly_aggregate_overlaid
from st_aggrid import AgGrid, GridOptionsBuilder
import st_aggrid as ag 

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


col1, col2 = st.columns([2,3])


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

st.write('---')

plot_container = st.container()
table_container = st.container()

fig = plot_monthly_aggregate_overlaid(london_monthly_aggregate, apulia_monthly_aggregate,
                            label1='London',
                            label2='Apulia',
                            feature_label=metric,
                            bar_width=0.35,
                            location='Comparison',
                            fig_width=900,
                            fig_height=500,
                            dark_mode=True,
                            show_title=False)

# ######
# # Summary Table Version 1
# ######

# with col2:
#     metric_column = metric
#     st.subheader(f"Monthly {metric_column}")
#     st.markdown("London (LND) vs Apulia (APL) Summary Table")
#     metric_dict = {
#         'Sum': 'month_sum',
#         'Average': 'month_avg',
#         'Median': 'month_median'
#     }

#     cols = ['month', metric_dict[metric], 'month_min', 'month_max']

#     centered_header_style = """
#     <style>
#     .ag-header-cell-label {
#         justify-content: center !important;
#         text-align: center !important;
#         white-space: normal !important;  /* wrap header text */
#     }
#     </style>
#     """

#       # Custom headers with line breaks
#     abbreviation_dict = {
#         'Average': 'Avg',
#         'Sum': 'Sum',
#         'Median': 'Med'
#     }

#     df_display = (
#     london_monthly_aggregate[cols]
#     .merge(apulia_monthly_aggregate[cols], on='month', suffixes=('_london', '_apulia'))
#     .rename(columns={
#         'month': 'Month',
#         f'{metric_dict[metric]}_london': f'LND | {abbreviation_dict[metric_column]}',
#         f'{metric_dict[metric]}_apulia': f'APL | {abbreviation_dict[metric_column]}',
#         'month_min_london': 'LND | Min',
#         'month_max_london': 'LND | Max',
#         'month_min_apulia': 'APL | Min',
#         'month_max_apulia': 'APL | Max'
#     })
#     )[['Month', f'LND | {abbreviation_dict[metric_column]}', f'APL | {abbreviation_dict[metric_column]}',
#        'LND | Min', 'APL | Min', 'LND | Max', 'APL | Max']]

#     london_color = "#0072B2"
#     apulia_color = "#E69F00"

#     highlight_min_style = {
#     'backgroundColor': '#d4f4dd',  # light green
#     'fontWeight': 'bold'
#     }

#     highlight_max_style = {
#     'backgroundColor': '#f9d4d4',  # light red
#     'fontWeight': 'bold'
#     }

#     custom_headers = {
#         f'London {abbreviation_dict[metric_column]}': f'LND | {abbreviation_dict[metric_column]}',
#         f'Apulia tt {metric_column}': f'APL | {metric_column}',
#         'London Min': 'LND | Min',
#         'Apulia Min': 'APL | Min',
#         'London Max': 'LND | Max',
#         'Apulia Max': 'APL | Max',
#         'Month': 'Month'
#     }
#         # GridOptionsBuilder
#     gb = GridOptionsBuilder.from_dataframe(df_display.reset_index(drop=True))

#     # # Override each column to disable filtering
#     for col in df_display.columns:
#         style = {}

#         if 'LND' in col:
#             style = {
#             'backgroundColor': london_color,
#             'color': 'white',
#             'fontWeight': 'bold'
#             }
#         elif 'APL' in col:
#             style = {
#             'backgroundColor': apulia_color,
#             'color': 'black',
#             'fontWeight': 'bold'
#             }

#         gb.configure_column(
#             col,
#             header_name=custom_headers.get(col, col),
#             filter=False,  # Explicitly disable filter here
#             cellStyle=style 
#         )

    
#     grid_options = gb.build()

#     st.markdown("""
#     <style>
#         .ag-header-cell-label {
#             justify-content: center !important;
#             text-align: center !important;
#             white-space: normal !important;
#             line-height: 1.2 !important;
#         }

#         .ag-header-cell-text {
#             word-wrap: break-word !important;
#             max-width: 100px !important;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#     # AgGrid display with auto height and dynamic sizing

#     AgGrid(
#         df_display,
#         gridOptions=grid_options,
#         theme='streamlit',
#         fit_columns_on_grid_load=True,
#         allow_unsafe_jscode=True,
#         height=400
#     )

# ######
# # Summary Table Version 2
# ######

metric_column = metric

metric_dict = {
'Sum': 'month_sum',
'Average': 'month_avg',
'Median': 'month_median'
}

abbreviation_dict = {
'Average': 'Avg',
'Sum': 'Sum',
'Median': 'Med'
}

cols = ['month', metric_dict[metric], 'month_min', 'month_max']

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

from st_aggrid.shared import JsCode
from st_aggrid import GridOptionsBuilder

gb = GridOptionsBuilder.from_dataframe(df_display.reset_index(drop=True))

# Pre-compute min/max per column
col_min_max = {
col: {
    "min": df_display[col].min(),
    "max": df_display[col].max()
}
for col in df_display.columns if col != "Month"
}

# V3
for col in df_display.columns:
    if col == "Month":
        gb.configure_column(col, header_name="Month", filter=False)
        continue

    # Determine base color style
    if 'LND' in col:
        base_bg = london_color
        font_color = 'white'
    elif 'APL' in col:
        base_bg = apulia_color
        font_color = 'black'
    else:
        base_bg = '#ffffff'
        font_color = 'black'

    # Pre-computed min/max
    col_min = col_min_max[col]["min"]
    col_max = col_min_max[col]["max"]

    # Overlay colors
    min_overlay = '#b6efc7'
    max_overlay = '#ffcccc'

    # JS for styling (background, bold)
    js_style = JsCode(f"""
    function(params) {{
        let style = {{
            backgroundColor: '{base_bg}',
            color: '{font_color}'
        }};
        if (params.value == {col_min}) {{
            style.backgroundColor = '{min_overlay}';
            style.fontWeight = 'bold';
        }} else if (params.value == {col_max}) {{
            style.backgroundColor = '{max_overlay}';
            style.fontWeight = 'bold';
        }}
        return style;
    }}
    """)

    # JS for icon prefix
    js_formatter = JsCode(f"""
    function(params) {{
        if (params.value == {col_min}) {{
            return '▼ ' + params.value;
        }} else if (params.value == {col_max}) {{
            return '▲ ' + params.value;
        }}
        return params.value;
    }}
    """)

    gb.configure_column(
        col,
        header_name=col,
        filter=False,
        cellStyle=js_style,
        valueFormatter=js_formatter
    )

grid_options = gb.build()

# Custom header style
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




col1, col2 = st.columns([3, 2])

with col1:
    with st.container():
        with st.spinner("Updating plot..."):
            st.subheader(f"{metric} Monthly Precipitation Comparison")
            st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container():
        with st.spinner("Generating summary table..."):
            st.subheader(f"Monthly {metric_column}")
            st.markdown("London (LND) vs Apulia (APL) Summary Table")

            AgGrid(
                df_display,
                gridOptions=grid_options,
                theme='streamlit',
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                height=400
            )
