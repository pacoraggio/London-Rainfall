import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

# Sample test data
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr"],
    "LND | Avg": [20.1, 15.2, 36.1, 28.3],
    "APL | Avg": [50.3, 162.8, 91.2, 60.1],
    "LND | Min": [5.1, 4.2, 7.5, 3.1],
    "APL | Min": [10.0, 15.0, 12.5, 11.0],
    "LND | Max": [50.1, 60.2, 58.3, 55.1],
    "APL | Max": [110.0, 162.8, 120.5, 130.1],
}

df_display = pd.DataFrame(data)

# Define colors
london_color = "#0072B2"
apulia_color = "#E69F00"

# GridOptionsBuilder setup
gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_default_column(filterable=False, sortable=True, resizable=True)

for col in df_display.columns:
    if col == "Month":
        gb.configure_column(col, header_name="Month", filter=False)
        continue

    min_val = round(float(df_display[col].min()), 6)
    max_val = round(float(df_display[col].max()), 6)

    # Assign base class for all cells in the column
    base_class = "lnd-cell" if "LND" in col else "apl-cell" if "APL" in col else ""

    # Define rules to apply "min-cell" or "max-cell" class
    gb.configure_column(
        col,
        header_name=col,
        filter=False,
        cellClass=base_class,
        cellClassRules={
            "min-cell": f"Number(params.value) === {min_val}",
            "max-cell": f"Number(params.value) === {max_val}"
        }
    )

grid_options = gb.build()

st.markdown("""
<style>
.lnd-cell {
    background-color: #0072B2 !important;
    color: white !important;
    font-weight: bold !important;
}
.apl-cell {
    background-color: #E69F00 !important;
    color: black !important;
    font-weight: bold !important;
}
.min-cell {
    background-color: #d4f4dd !important;
    font-weight: bold !important;
}
.max-cell {
    background-color: #f9d4d4 !important;
    font-weight: bold !important;
}
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

st.subheader("🔬 AgGrid Min/Max Highlight Test")

AgGrid(
    df_display,
    gridOptions=grid_options,
    theme='material',  # ✅ better CSS compatibility
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    height=400,
    domLayout='autoHeight'
)