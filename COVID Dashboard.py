import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --------
PRIMARY = "#1B3A4B"   # dark slate blue (the main colour)
ACCENT = "#E07A5F"    # terracotta (used to highlight the key point)

st.set_page_config(page_title="COVID-19: A Pandemic of Inequalities",
                   layout="wide")


# =========================================================== #
# 1. DATA
# =========================================================== #

# ---- Cases and deaths over time (monthly totals) ----
daily_raw = pd.read_csv("WHO COVID-19 Daily Data.csv")
daily_raw["Date_reported"] = pd.to_datetime(daily_raw["Date_reported"])
daily = (
    daily_raw.groupby(pd.Grouper(key="Date_reported", freq="MS"))[["New_cases", "New_deaths"]]
    .sum().reset_index()
    .rename(columns={"Date_reported": "Date", "New_cases": "Cases", "New_deaths": "Deaths"})
)

# ---- Deaths by age group and income level (each income column adds to ~100%) ----
age_raw = pd.read_csv("WHO COVID-19 Monthly Deaths by Age.csv")
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
# 2. SIDEBAR  (menu + filters)
# =========================================================== #
st.sidebar.title("🦠 Pandemic of Inequalities")
page = st.sidebar.radio("Go to:", [
    "1. Introduction",
    "2. Cases and deaths over time",
    "3. Which ages died",
    "4. Vaccines and income",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Year filter (used on pages 1 and 2)
years = daily["Date"].dt.year
y_min, y_max = int(years.min()), int(years.max())
year_range = st.sidebar.slider("Year range", y_min, y_max, (y_min, y_max))

# Income filter (used on pages 3 and 4)
all_income = ["Low", "Lower-middle", "Upper-middle", "High"]
chosen_income = st.sidebar.multiselect("Income groups", all_income, default=all_income)
if not chosen_income:                       # never let it be empty
    chosen_income = all_income

# Apply the year filter once
daily_f = daily[(years >= year_range[0]) & (years <= year_range[1])]


# =========================================================== #
# 3. PAGES
# =========================================================== #

# ---- Page 1: Introduction ----
if page.startswith("1"):
    st.title("COVID-19: A Pandemic of Inequalities")
    st.write(
        "The pandemic hit everyone, but not equally. This dashboard looks "
        "at two kinds of inequality: **age** and **income**. Use the filters "
        "in the sidebar to explore the numbers."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total cases", f"{daily_f['Cases'].sum() / 1e6:.0f} M")
    c2.metric("Total deaths", f"{daily_f['Deaths'].sum() / 1e6:.1f} M")
    c3.metric("Years shown", f"{year_range[0]}–{year_range[1]}")


# ---- Page 2: Cases and deaths over time (Plotly) ----
elif page.startswith("2"):
    st.subheader("Two Curves, One Pandemic")
    measure = st.radio("Choose a measure:", ["Cases", "Deaths"], horizontal=True)
    st.caption(f"Monthly new {measure.lower()}, {year_range[0]}–{year_range[1]}")

    color = PRIMARY if measure == "Cases" else ACCENT
    fig = px.line(daily_f, x="Date", y=measure)
    fig.update_traces(line_color=color)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See the monthly numbers"):
        st.dataframe(daily_f, use_container_width=True)


# ---- Page 3: Which ages died (Matplotlib + Seaborn) ----
elif page.startswith("3"):
    st.subheader("The Pandemic Aged Upward")
    st.caption("Average share of deaths by age group, for the chosen income groups")

    avg = age.set_index("Age")[chosen_income].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(avg.index, avg.values, color=PRIMARY, marker="o", linewidth=2)
    ax.plot(avg.index[2:], avg.values[2:], color=ACCENT, marker="o", linewidth=2)
    ax.set_ylabel("Share of deaths (%)")
    st.pyplot(fig)
    st.write("Most deaths were in people **65 and older** (orange).")

    st.subheader("Where Age Meets Income")
    st.caption("Share of each income group's deaths, by age")
    heat = age.set_index("Age")[chosen_income]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat, annot=True, fmt="d",
                cmap=sns.light_palette(PRIMARY, as_cmap=True), ax=ax2)
    st.pyplot(fig2)


# ---- Page 4: Vaccines and income (Seaborn) ----
elif page.startswith("4"):
    st.subheader("The Vaccine Divide")
    st.caption("Vaccination coverage by income level")

    order = [g for g in all_income if g in chosen_income]
    vax_f = vax[vax["Income"].isin(order)]
    colors = {x: (ACCENT if x == "Low" else PRIMARY) for x in order}

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=vax_f, x="Income", y="Coverage", order=order,
                hue="Income", palette=colors, legend=False, ax=ax)
    ax.set_ylabel("Fully vaccinated per 100 people")
    st.pyplot(fig)

    with st.expander("See the median coverage per group"):
        st.dataframe(
            vax_f.groupby("Income")["Coverage"].median().reindex(order).round(1),
            use_container_width=True,
        )

    st.write(
        "**Conclusion.** To prepare for the next pandemic: protect the "
        "elderly, share vaccines more fairly, and help low-income countries "
        "count deaths better."
    )
