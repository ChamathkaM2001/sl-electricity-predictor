"""Sri Lanka Household Electricity Predictor — Streamlit app for the NGBoost model."""
import pickle

import numpy as np
import pandas as pd
import altair as alt
import shap
import streamlit as st
from scipy.stats import norm

import ngb_compat  # noqa: F401  (needed so pickle can find the NGBoostRegressor class)

st.set_page_config(page_title="Sri Lanka Electricity Predictor", page_icon="⚡", layout="wide")


# ---------- load model once ----------
@st.cache_resource
def load_bundle():
    with open("electricity_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    pre = bundle["pipeline"]["pre"]
    ngb = bundle["pipeline"]["model"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    explainer = shap.TreeExplainer(ngb, model_output=0)
    return bundle, pre, ngb, names, explainer


bundle, pre, ngb, names, explainer = load_bundle()
FEATURES, DEFAULTS, DEMO = bundle["features"], bundle["defaults"], bundle["demo"]
Z80 = norm.ppf(0.90)

LABELS = {
    "n_members": "Household members", "n_ac": "Air conditioners", "ac_hours_week": "AC hours per week",
    "n_fans": "Fans", "fan_hours_week": "Fan hours per week", "n_fridge": "Refrigerators",
    "n_washing_machine": "Washing machines", "n_appliances_total": "Total appliances",
    "n_bulbs": "Light bulbs", "floor_area": "Floor area", "n_rooms": "Rooms",
    "sec_class": "Socio-economic class", "monthly_expenditure": "Monthly expenditure",
    "has_solar": "Rooftop solar", "cook_electric": "Electric cooking", "cook_firewood": "Firewood cooking",
    "lighting_kwh_week": "Lighting energy", "bulb_hours_week": "Bulb hours", "n_meters": "Electricity meters",
    "avg_hours_at_home": "Hours at home", "n_bedrooms": "Bedrooms",
}


def nice(name):
    for prefix in ("house_type_", "water_heating_", "area_", "built_year_", "wall_material_", "roof_material_"):
        if name.startswith(prefix):
            return f"{prefix[:-1].replace('_', ' ').capitalize()}: {name[len(prefix):].replace('_', ' ')}"
    if name in LABELS:
        return LABELS[name]
    if name.startswith("n_"):
        return "Number of " + name[2:].replace("_", " ") + "s"
    return name.replace("_", " ").capitalize()


def predict(row):
    X = pd.DataFrame([row])[FEATURES]
    Xt = pre.transform(X)
    dist = ngb.pred_dist(Xt)
    mu, sigma = dist.params["loc"][0], dist.params["scale"][0] * bundle["scale_factor"]
    return np.expm1(mu), np.expm1(mu - Z80 * sigma), np.expm1(mu + Z80 * sigma), Xt


def tariff_block(kwh):
    for upper, label in [(60, "0–60"), (90, "61–90"), (120, "91–120"), (180, "121–180")]:
        if kwh <= upper:
            return f"{label} units"
    return "181+ units"


# ---------- sidebar inputs ----------
st.sidebar.header("🏠 Describe the household")
use_demo = st.sidebar.checkbox(f"Load demo household {DEMO['household_ID']} (never seen by the model)")
start = DEMO if use_demo else DEFAULTS
opts = bundle["cat_options"]


def num(label, key, lo, hi, step=1, help=None):
    value = float(start[key]) if not pd.isna(start[key]) else float(DEFAULTS[key])
    value = min(max(value, lo), hi)
    if step == 1:
        return st.sidebar.number_input(label, int(lo), int(hi), int(round(value)), 1, help=help, key=f"{key}_{use_demo}")
    return st.sidebar.number_input(label, float(lo), float(hi), float(value), float(step), help=help, key=f"{key}_{use_demo}")


def pick(label, key):
    options = opts[key]
    return st.sidebar.selectbox(label, options, index=options.index(start[key]) if start[key] in options else 0,
                                key=f"{key}_{use_demo}")


def yesno(label, key):
    return int(st.sidebar.checkbox(label, value=bool(start[key]), key=f"{key}_{use_demo}"))


row = {f: start[f] for f in FEATURES}   # features not on the form keep the demo/default value
st.sidebar.subheader("People and house")
row["n_members"] = num("Household members", "n_members", 1, 15)
row["area"] = pick("Electricity area (CSC)", "area")
row["house_type"] = pick("House type", "house_type")
row["floor_area"] = num("Floor area (sq ft)", "floor_area", 100, 9000, 50.0)
row["n_rooms"] = num("Total rooms (incl. kitchen, bathrooms)", "n_rooms", 1, 40)
row["sec_class"] = st.sidebar.select_slider("Socio-economic class", options=[1, 2, 3, 4, 5],
                                            value=int(start["sec_class"]), format_func=lambda v: "EDCBA"[v - 1],
                                            key=f"sec_{use_demo}")

st.sidebar.subheader("Cooling")
row["n_ac"] = num("Air conditioners", "n_ac", 0, 10)
row["ac_hours_week"] = num("Total AC hours per week (all ACs)", "ac_hours_week", 0, 500, 0.5)
row["n_fans"] = num("Fans", "n_fans", 0, 30)
row["fan_hours_week"] = num("Total fan hours per week (all fans)", "fan_hours_week", 0, 2000, 0.5)

st.sidebar.subheader("Appliances and energy")
row["n_fridge"] = num("Refrigerators", "n_fridge", 0, 5)
row["n_washing_machine"] = num("Washing machines", "n_washing_machine", 0, 3)
row["n_appliances_total"] = num("Total electrical appliances (incl. phones, TVs…)", "n_appliances_total", 0, 80,
                                help="Count every electrical appliance, including the fridges and washing machines above.")
row["n_bulbs"] = num("Light bulbs", "n_bulbs", 0, 100)
row["water_heating"] = pick("Water heating for bathing", "water_heating")
row["cook_electric"] = yesno("Cooks with grid electricity", "cook_electric")
row["cook_firewood"] = yesno("Cooks with firewood", "cook_firewood")
row["has_solar"] = yesno("Has rooftop solar", "has_solar")

# ---------- main panel ----------
st.title("⚡ Sri Lanka Household Electricity Predictor")
st.write("Estimates a household's **average monthly grid electricity consumption** from its house, people and "
         "appliances, with an honest uncertainty range. Model: **NGBoost** trained on about 3,200 real Sri Lankan "
         "households (RECON-SL, Oct 2022 – Oct 2024).")

pred, lo, hi, Xt = predict(row)
c1, c2, c3 = st.columns(3)
c1.metric("Predicted consumption", f"{pred:.0f} units / month")
c2.metric("80% likely range", f"{lo:.0f} – {hi:.0f} units")
c3.metric("Tariff block", tariff_block(pred))
if use_demo:
    st.info(f"Actual average consumption of {DEMO['household_ID']}: **{DEMO['avg_kwh']:.0f} units / month**")
st.caption("1 unit = 1 kWh. The range is calibrated: about 8 in 10 real households fall inside their 80% range. "
           "Tariffs change often, so check the current CEB/LECO rates to convert units into a bill.")

# SHAP explanation
st.subheader("🔍 What drives this prediction?")
sv = explainer(pd.DataFrame(Xt, columns=names))
contrib = pd.Series(sv.values[0], index=names)
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])
pct = (np.exp(top) - 1) * 100
chart_df = pd.DataFrame({"Feature": [nice(n) for n in top.index], "Effect (%)": pct.values.round(1),
                         "Direction": np.where(pct.values >= 0, "Raises", "Lowers")})
chart = (alt.Chart(chart_df).mark_bar()
         .encode(x=alt.X("Effect (%):Q", title="Effect on prediction (%)"),
                 y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                 color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises", "Lowers"],
                                 range=["#d62728", "#1f77b4"]), legend=alt.Legend(title=None, orient="bottom")),
                 tooltip=["Feature", "Effect (%)"])
         .properties(height=36 * len(chart_df)))
st.altair_chart(chart, use_container_width=True)
base = np.expm1(sv.base_values[0])
st.caption(f"Each bar shows how much a feature pushes this household above (+) or below (−) an average home "
           f"({base:.0f} units). Computed with SHAP.")

# what-if scenarios
st.subheader("💡 What if…?")
scenarios = {
    "Halve AC use": {"ac_hours_week": row["ac_hours_week"] / 2},
    "Remove all ACs": {"n_ac": 0, "ac_hours_week": 0},
    "Halve fan use": {"fan_hours_week": row["fan_hours_week"] / 2},
    "One fewer refrigerator": {"n_fridge": max(row["n_fridge"] - 1, 0),
                               "n_appliances_total": max(row["n_appliances_total"] - (row["n_fridge"] > 0), 0)},
    "Two more family members": {"n_members": row["n_members"] + 2},
}
rows = []
for name, change in scenarios.items():
    new_pred = predict({**row, **change})[0]
    rows.append({"Scenario": name, "Predicted units": round(new_pred), "Change": f"{new_pred - pred:+.0f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("About the model"):
    m = bundle["metrics"]
    st.write(f"- Test-set mean absolute error: **{m['test_mae_kwh']:.1f} units**, median error **{m['test_median_ae_kwh']:.1f} units**\n"
             f"- 80% ranges contained the true value for **{m['coverage_80']:.0%}** of test households\n"
             "- Most reliable for typical homes (60–180 units); very low and very high users are pulled towards "
             "the middle\n- The model knows what a household owns better than how intensively it uses it")
