import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

PRIMARY = "#1B3A4B"
ACCENT = "#E07A5F"

# =========================================================== #
# 1. LOAD AND PREPARE DATA
# =========================================================== #
daily_raw = pd.read_csv("WHO COVID-19 Daily Data.csv")
daily_raw["Date_reported"] = pd.to_datetime(daily_raw["Date_reported"])
daily = (
    daily_raw.groupby(pd.Grouper(key="Date_reported", freq="MS"))[["New_cases", "New_deaths"]]
    .sum().reset_index()
    .rename(columns={"Date_reported": "Date", "New_cases": "Cases", "New_deaths": "Deaths"})
)

# ---- Total deaths by country, for the map ----
age_raw = pd.read_csv("WHO COVID-19 Monthly Deaths by Age.csv")
country = (
    age_raw.groupby("Country_code")["Deaths"].sum().reset_index()
    .rename(columns={"Country_code": "Code"})
)
country = country[country["Code"].str.fullmatch(r"[A-Z]{3}").fillna(False)]  # drop non-country codes

# ---- Deaths by age group and income level (each income column adds to ~100%) ----
income_names = {"LIC": "Low", "LMC": "Lower-middle", "UMC": "Upper-middle", "HIC": "High"}
age_names = {"0_4": "0-4", "5_14": "5-14", "15_64": "15-64", "65+": "65+"}
age = (
    age_raw.pivot_table(index="Agegroup", columns="Wb_income", values="Deaths", aggfunc="sum")
    .reindex(["0_4", "5_14", "15_64", "65+"]).fillna(0).rename(index=age_names)
)
age = (age / age.sum() * 100).round().astype(int)
age = age.rename(columns=income_names)[["Low", "Lower-middle", "Upper-middle", "High"]]
age = age.reset_index().rename(columns={"Agegroup": "Age"})

# ---- Vaccination coverage (at least one dose, per 100) by income level ----
vax_raw = pd.read_csv("COVID Vaccine Uptake 2021-2023.csv")
coverage = vax_raw.groupby("COUNTRY")["COVID_VACCINE_COV_TOT_A1D"].max()
income_by_country = age_raw.drop_duplicates("Country_code").set_index("Country_code")["Wb_income"]
vax = pd.DataFrame({"Coverage": coverage})
vax["Income"] = vax.index.map(income_by_country).map(income_names)
vax = vax.dropna(subset=["Coverage", "Income"]).reset_index(drop=True)

# =========================================================== #
# 2. SIDEBAR MENU
# =========================================================== #
st.sidebar.title("🦠 Pandemic of Inequalities")
page = st.sidebar.radio("Go to:", [
    "1. Introduction",
    "2. Cases and deaths over time",
    "3. Where deaths happened",
    "4. Which ages died",
    "5. Vaccines and income",
])


# =========================================================== #
# 3. PAGES
# =========================================================== #

# ---- Page 1: Introduction ----
if page.startswith("1"):
    st.title("COVID-19: A Pandemic of Inequalities")
    st.write(
        "The pandemic hit everyone, but not equally. This dashboard looks "
        "at two kinds of inequality: **age** and **income**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total cases", f"{daily['Cases'].sum() / 1e6:.0f} M")
    c2.metric("Total deaths", f"{daily['Deaths'].sum() / 1e6:.1f} M")
    c3.metric("Countries shown", len(country))


# ---- Page 2: Cases and deaths over time (Plotly) ----
elif page.startswith("2"):
    st.subheader("Two Curves, One Pandemic")
    st.caption("New cases over time")
    fig = px.line(daily, x="Date", y="Cases")
    fig.update_traces(line_color=PRIMARY)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("And the Cost in Lives")
    st.caption("New deaths over time")
    fig2 = px.line(daily, x="Date", y="Deaths")
    fig2.update_traces(line_color=ACCENT)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Cases and deaths came in waves.")


# ---- Page 3: Where deaths happened (Plotly map) ----
elif page.startswith("3"):
    st.subheader("The Geography of Loss")
    st.caption("Total reported deaths by country")
    fig = px.choropleth(
        country, locations="Code", color="Deaths",
        color_continuous_scale=["#E8EEF1", PRIMARY],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("A few large countries account for most recorded death")


# ---- Page 4: Which ages died (Matplotlib + Seaborn) ----
elif page.startswith("4"):
    st.subheader("The Pandemic Aged Upward")
    st.caption("Average share of deaths by age group")
    age["Average"] = age[["Low", "Lower-middle", "Upper-middle", "High"]].mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(age["Age"], age["Average"], color=PRIMARY, marker="o", linewidth=2)
    # highlight the two oldest groups in the accent colour
    ax.plot(age["Age"][3:], age["Average"][3:], color=ACCENT, marker="o", linewidth=2)
    ax.set_ylabel("Share of deaths (%)")
    st.pyplot(fig)
    st.info("Most deaths were in people 65 and older (orange)")

    st.subheader("Where Age Meets Income")
    st.caption("Share of each income group's deaths, by age")
    heat = age.set_index("Age")[["Low", "Lower-middle", "Upper-middle", "High"]]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat, annot=True, fmt="d",
                cmap=sns.light_palette(PRIMARY, as_cmap=True), ax=ax2)
    st.pyplot(fig2)
    st.info("The old-age pattern shows up in every income group.")


# ---- Page 5: Vaccines and income (Seaborn) ----
elif page.startswith("5"):
    st.subheader("The Vaccine Divide")
    st.caption("Vaccination coverage by income level")
    order = ["Low", "Lower-middle", "Upper-middle", "High"]
    colors = {x: (ACCENT if x == "Low" else PRIMARY) for x in order}

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=vax, x="Income", y="Coverage", order=order,
                hue="Income", palette=colors, legend=False, ax=ax)
    ax.set_ylabel("Fully vaccinated per 100 people")
    st.pyplot(fig)
    st.info("Low-income countries (orange) vaccinated far fewer people.")

    st.write(
        "**Conclusion.** To prepare for the next pandemic: protect the "
        "elderly, share vaccines more fairly, and help low-income countries "
        "count deaths better."
    )
