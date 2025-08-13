# Read Me

London-Rainfall is the repository of a project analysing London precipitation and comparing it with the Italian Apulia region.

## Files and folders
- notebook
  - `how_to_test_a_feeling_rainfall_analysis.ipynb`. It contains the information about the project explaining why it was created, history of different adopted strategies throughtout the analysis and deeper explaination about the folders structures
- folders
  - `grib` contains the ERA5 downloaded files in `grib` format. These files are then transformed and saved in `.csv` files in the `data` folder to be further processed in the different notebooks and streamlit application.
  - `data` contains the original `.grib` data files downloaded that will be processed and transformed in `.csv` files to perform the different analysis. **THE FOLDER IS NOT UPLOADED ON GITHUB FOR MEMORY CONSTRAINS**.
  - `src` contains the source code of python functions used throughout the project.
  - `notebooks`
  - `streamlit` contains a strealit app files built on top of the project.

###  Still to Modify
- supporting notebooks:
  - data_source_analysis_and_manipulation
  - heathrow_data
  - data_visualization
    - `plot_rainfall` barplot of yearly aggregate measure (e.g. sum, mean, median)
    - `plot_min_max_rainfall` lineplot of yearly monthly min and max values    

The structure of subfolder is illustrated in notebook inside the folders

 - `output` contains the `.parquet` files resulting from data wrangling and analysis process. This folder is uploaded on GitHub (WIP).
