
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#--------
PRIMARY = "#1B3A4B"   # dark slate blue
ACCENT = "#E07A5F"    # terracotta 

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

# ---- Deaths by age and income ----
age_raw = pd.read_csv("WHO COVID-19 Monthly Deaths by Age.csv")
income_names = {"LIC": "Low", "LMC": "Lower-middle", "UMC": "Upper-middle", "HIC": "High"}
age_names = {"0_4": "0-4", "5_14": "5-14", "15_64": "15-64", "65+": "65+"}
order_age = ["0-4", "5-14", "15-64", "65+"]
order_income = ["Low", "Lower-middle", "Upper-middle", "High"]

# share of deaths by age group (overall)
deaths_by_age = (age_raw.groupby("Agegroup")["Deaths"].sum()
                 .reindex(["0_4", "5_14", "15_64", "65+"]).rename(index=age_names))
age_share = (deaths_by_age / deaths_by_age.sum() * 100)

# share of recorded deaths by income level
deaths_by_income = (age_raw.groupby("Wb_income")["Deaths"].sum()
                    .rename(index=income_names).reindex(order_income))
income_share = (deaths_by_income / deaths_by_income.sum() * 100)

# age x income table (for the heatmap)
heat = (age_raw.pivot_table(index="Agegroup", columns="Wb_income", values="Deaths", aggfunc="sum")
        .reindex(["0_4", "5_14", "15_64", "65+"]).fillna(0).rename(index=age_names))
heat = (heat / heat.sum() * 100).round().astype(int)
heat = heat.rename(columns=income_names)[order_income]
heat.index.name = "Age"

# ---- Vaccination coverage by income ----
vax_raw = pd.read_csv("COVID Vaccine Uptake 2021-2023.csv")
coverage = vax_raw.groupby("COUNTRY")["COVID_VACCINE_COV_TOT_A1D"].max()
income_by_country = age_raw.drop_duplicates("Country_code").set_index("Country_code")["Wb_income"]
vax = pd.DataFrame({"Coverage": coverage})
vax["Income"] = vax.index.map(income_by_country).map(income_names)
vax = vax.dropna(subset=["Coverage", "Income"]).reset_index(drop=True)


# =========================================================== #
# 2. SIDEBAR
# =========================================================== #
st.sidebar.title("🦠 Pandemic of Inequalities")
page = st.sidebar.radio("Read the story:", [
    "1. Introduction",
    "2. The pandemic over time",
    "3. Who it hit hardest",
    "4. The vaccine divide",
])

st.sidebar.markdown("---")
years = daily["Date"].dt.year
y_min, y_max = int(years.min()), int(years.max())
year_range = st.sidebar.slider("Year range", y_min, y_max, (y_min, y_max))
daily_f = daily[(years >= year_range[0]) & (years <= year_range[1])]


# small helper for the lollipop charts (used on page 3)
def lollipop(series, highlight, xlabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, value) in enumerate(series.items()):
        col = ACCENT if label == highlight else PRIMARY
        ax.hlines(i, 0, value, color=col, alpha=0.5, linewidth=2.5)
        ax.plot(value, i, "o", color=col, markersize=12)
        label_txt = f"{value:.0f}%" if value >= 1 else f"{value:.1f}%"
        ax.text(value + series.max() * 0.02, i, label_txt, va="center")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series.index)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, series.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    return fig


# =========================================================== #
# 3. PAGES
# =========================================================== #

# ---- Page 1: Introduction ----
if page.startswith("1"):
    st.title("COVID-19: A Pandemic of Inequalities")
    st.write(
        "COVID-19 was a global pandemic, but the burden was never shared "
        "equally. This is a short data story about **who died** and **who "
        "got protected** — and the gap between them."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Reported cases", f"{daily_f['Cases'].sum() / 1e6:.0f} M")
    c2.metric("Reported deaths", f"{daily_f['Deaths'].sum() / 1e6:.1f} M")
    c3.metric("Years shown", f"{year_range[0]}–{year_range[1]}")



# ---- Page 2: The pandemic over time (Plotly) ----
elif page.startswith("2"):
    st.subheader("Two Curves, One Pandemic")
    measure = st.radio("Choose a measure:", ["Cases", "Deaths"], horizontal=True)
    st.caption(f"Monthly new {measure.lower()}, {year_range[0]}–{year_range[1]}")

    color = PRIMARY if measure == "Cases" else ACCENT
    fig = px.line(daily_f, x="Date", y=measure)
    fig.update_traces(line_color=color)
    st.plotly_chart(fig, use_container_width=True)
    st.write("The virus arrived in waves — each one a fresh surge of illness "
             "and loss before the next began.")


# ---- Page 3: Who it hit hardest (Matplotlib + Seaborn) ----
elif page.startswith("3"):
    st.subheader("Who the Pandemic Hit Hardest")
    lens = st.radio("Look at the deaths by:", ["Age", "Income"], horizontal=True)

    if lens == "Age":
        st.caption("Share of recorded deaths by age group")
        st.pyplot(lollipop(age_share, "65+", "Share of deaths (%)"))
        st.write(
            f"The pandemic fell hardest on the old: people **65 and older** "
            f"(orange) account for about **{age_share['65+']:.0f}%** of "
            "recorded deaths."
        )
    else:
        st.caption("Share of recorded deaths by country income level")
        st.pyplot(lollipop(income_share, "Low", "Share of deaths (%)"))
        low_txt = "less than 1%" if income_share["Low"] < 1 else f"{income_share['Low']:.0f}%"
        st.write(
            f"Low-income countries (orange) account for only "
            f"**{low_txt}** of recorded deaths — not because fewer people "
            "died there, but because deaths are far harder to count where "
            "health systems are stretched. The real toll is almost certainly "
            "higher than the records show."
        )

    st.subheader("Where Age Meets Income")
    st.caption("Share of each income group's deaths, by age")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(heat, annot=True, fmt="d",
                cmap=sns.light_palette(PRIMARY, as_cmap=True), ax=ax2)
    ax2.set_xlabel("")
    st.pyplot(fig2)


# ---- Page 4: The vaccine divide (Seaborn) ----
elif page.startswith("4"):
    st.subheader("The Vaccine Divide")
    st.caption("Almost everyone in wealthy countries got a shot. The poorest "
               "were left behind.")

    med = vax.groupby("Income")["Coverage"].median()
    gap = med["High"] - med["Low"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Low-income (median)", f"{med['Low']:.0f} / 100")
    c2.metric("High-income (median)", f"{med['High']:.0f} / 100")
    c3.metric("The gap", f"{gap:.0f} points")

    st.markdown("**Set a vaccination target and see who reached it:**")
    target = st.slider("Doses per 100 people", 0, 100, 70)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.stripplot(data=vax, x="Income", y="Coverage", order=order_income,
                  color=PRIMARY, alpha=0.6, jitter=0.2, size=6, ax=ax)
    ax.axhline(target, color=ACCENT, linestyle="--", linewidth=2)
    ax.text(3.4, target + 1.5, f"target: {target}", color=ACCENT, ha="right")
    ax.set_ylabel("Fully vaccinated per 100 people")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

    reached = vax[vax["Coverage"] >= target].groupby("Income").size()
    totals = vax.groupby("Income").size()
    lr, lt = int(reached.get("Low", 0)), int(totals.get("Low", 0))
    hr, ht = int(reached.get("High", 0)), int(totals.get("High", 0))
    st.write(
        f"At a target of **{target} doses per 100 people**, only "
        f"**{lr} of {lt}** low-income countries reached it — compared with "
        f"**{hr} of {ht}** high-income countries."
    )
