from streamlit_option_menu import option_menu
import streamlit as st

from src.annual_precipitation import run_annual_precipitation
from src.monthly_precipitation import run_monthly_precipitation



st.set_page_config(
    page_title="London and Apulia Rainfall Analysis",
    page_icon="🌧️",
    layout="wide",  # This sets wide mode
    initial_sidebar_state="expanded"
)

selected = option_menu(
    None, ["Home", "Annual Precipitation Analysis", "Monthly Precipitation Analysis"],
    icons=["house", "calendar3", "bar-chart"],
    menu_icon="cast", default_index=0, orientation="horizontal"
)

if selected == "Home":
    st.title("🌧️ Where does it rain more? London or Apulia?")
    st.markdown("---")

    st.markdown(
        "Lately, I was chatting on WhatsApp with a friend of mine. Since we live in different countries, "
        "the weather is always a topic we use to catch up (*\"here it’s sunny\"*, *\"here it’s raining and miserable\"*, "
        "*\"there are no middle seasons anymore\"*). "
        "During the chat, out of the blue, my friend claimed that in Apulia it rains more than in London, but less often. "
        "As he is usually right—and I usually try to prove him wrong—I started wondering how I could actually verify this. "
        "Most people would just ask ChatGPT, Claude, or Google to settle the argument. "
        "But I was curious about how I could **build the answer myself**, by collecting and analysing real data."
    )

    st.markdown(
        "The result is this Streamlit app, which processes and visualizes rainfall data extracted from the "
        "[ERA5-Land reanalysis dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview). "
        "And the answer is clear:"
    )

    st.markdown("***NO, it rains more in London than in Apulia.***")

    st.markdown(
        "I downloaded ERA5-Land data for the two regions—London and Apulia—and processed it to compare annual and monthly precipitation. "
        "Looking at the period from **January 2000 to June 2025**, London receives on average **77 mm more rainfall per year** than Apulia. "
        "In London, rainfall is more evenly distributed across the year, whereas Apulia shows a sharp summer drop, "
        "with **January and October** standing out as the wettest months."
    )

    st.markdown(
        "This app allows you to explore the data interactively, breaking it down into annual and monthly analyses, "
        "and highlighting the differences in rainfall patterns between the two regions. "
        "You can select specific years to see how rainfall varies over time, "
        "and explore rainfall distribution on maps for a geographical perspective in the annual analysis."
    )


    with st.expander("About the ERA5-Land Dataset"):
        st.markdown(
            "This application uses rainfall data from the [ERA5-Land reanalysis dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview). "
            "ERA5-Land is a global climate reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) "
            "and made available through the Copernicus Climate Data Store. "
            "It provides high-resolution (9 km) hourly estimates of surface variables such as temperature, soil moisture, and precipitation, "
            "covering the period from 1950 to the present."
        )
        st.markdown(
            "Unlike traditional weather station records, which can be limited in coverage and continuity, "
            "ERA5-Land blends observations from satellites, ground stations, and other instruments with advanced numerical weather models. "
            "This produces a consistent and gap-free dataset, particularly valuable for regional climate analysis."
        )
        st.markdown(
            "For this project, hourly precipitation data were aggregated into monthly and annual totals, "
            "allowing for direct comparison of seasonal cycles and long-term rainfall trends between London and Apulia. "
            "This approach captures both inter-annual variability and differences between temperate and Mediterranean rainfall regimes."
        )
        st.markdown(
            "**Advantages of ERA5-Land include:**  \n"
            "- High spatial and temporal resolution: hourly data on a fine 9 km grid enables precise regional analysis.  \n"
            "- Consistency over time: reanalysis ensures a uniform methodology, avoiding inconsistencies in raw observations.  \n"
            "- Open access: ERA5-Land is freely available, making climate-quality data accessible to researchers, policymakers, and the public."
        )
        st.markdown(
            "By leveraging these data, this app provides an evidence-based overview of rainfall variability "
            "in two regions with very different hydrological and climatic characteristics: "
            "the **temperate London area** and the **Mediterranean Apulia region**."
        )
elif selected == "Annual Precipitation Analysis":
    run_annual_precipitation()
elif selected == "Monthly Precipitation Analysis":
    run_monthly_precipitation()