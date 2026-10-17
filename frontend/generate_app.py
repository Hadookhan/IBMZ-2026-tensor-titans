# =========================================================
# HOMEmatch AI
# Two-column / "transposed" Jupyter frontend
# =========================================================

import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.graph_objects as go

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, clear_output


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "../data/property_price_data.csv"
MODEL_PATH = "../models/future_growth_rf_v1.pkl"

PROPERTYDATA_BASE_URL = "https://api.propertydata.co.uk"
PROPERTYDATA_API_KEY = os.environ.get("PROPERTYDATA_API_KEY")


# These MUST be the same features used to train the model.
model_features = [
    "price_data_1",
    "price_data_2",
    "price_data_3",
    "price_data_4",
    "price_data_5",
    "price_data_6",
    "growth_data_2",
    "growth_data_3",
    "growth_data_4",
    "growth_data_5",
    "growth_data_6"
]


# =========================================================
# CHECK API KEY
# =========================================================

if not PROPERTYDATA_API_KEY:
    print(
        "WARNING: PROPERTYDATA_API_KEY is not set.\n"
        "Run this in a separate cell:\n\n"
        'os.environ["PROPERTYDATA_API_KEY"] = "YOUR_TOKEN"'
    )


# =========================================================
# LOAD DATA + MODEL
# =========================================================

df = pd.read_csv(DATA_PATH)

growth_model = joblib.load(MODEL_PATH)

print(f"Loaded {len(df):,} candidate areas")
print("Growth model loaded successfully.")


# =========================================================
# PROPERTYDATA API CLIENT
# =========================================================

def propertydata_request(endpoint, **params):

    if not PROPERTYDATA_API_KEY:
        raise RuntimeError(
            "PROPERTYDATA_API_KEY has not been set."
        )

    url = f"{PROPERTYDATA_BASE_URL}/{endpoint}"

    headers = {
        "X-API-Key": PROPERTYDATA_API_KEY
    }

    for attempt in range(3):

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        # Rate limit
        if response.status_code == 429:

            wait_seconds = int(
                response.headers.get(
                    "Retry-After",
                    5
                )
            )

            time.sleep(wait_seconds)
            continue

        try:
            data = response.json()

        except Exception:
            raise RuntimeError(
                f"Invalid response from {endpoint}: "
                f"{response.text[:300]}"
            )

        if response.ok:

            if data.get("status") == "success":
                return data

            raise RuntimeError(
                f"PropertyData error: "
                f"{data.get('code')} - "
                f"{data.get('message')}"
            )

        raise RuntimeError(
            f"HTTP {response.status_code}: {data}"
        )

    raise RuntimeError(
        "PropertyData request repeatedly throttled."
    )

# tests = [
#     ("crime", "BB12"),
#     ("schools", "BB12"),
#     ("crime", "NE1"),
#     ("schools", "NE1"),
#     ("crime", "B10"),
#     ("schools", "B10"),
# ]

# for endpoint, postcode in tests:

#     print(
#         "\n",
#         "=" * 50,
#         f"\n{endpoint.upper()} - {postcode}"
#     )

#     try:

#         result = propertydata_request(
#             endpoint,
#             postcode=postcode
#         )

#         print(result)

#     except Exception as e:

#         print("ERROR:", e)


# =========================================================
# POSTCODE HELPER
# =========================================================

# def postcode_district(postcode):

#     return (
#         str(postcode)
#         .strip()
#         .upper()
#         .split()[0]
#     )


# =========================================================
# PROPERTYDATA FUNCTIONS
# Cached so repeat searches do not call the API again
# =========================================================

@lru_cache(maxsize=500)
def fetch_live_price(postcode, bedrooms=0):

    params = {
        "postcode": postcode
    }

    if bedrooms:
        params["bedrooms"] = bedrooms

    result = propertydata_request(
        "prices",
        **params
    )

    return float(
        result["data"]["average"]
    )


@lru_cache(maxsize=500)
def fetch_crime(postcode):

    result = propertydata_request(
        "crime",
        postcode=postcode
    )

    crime_rate = result.get("crimes_per_thousand")
    crime_rating = result.get("crime_rating")

    if crime_rate is None:
        raise RuntimeError(
            f"No crime data returned for {postcode}. "
            f"Response: {result}"
        )

    return {
        "crime_rate": float(crime_rate),
        "crime_rating": crime_rating or "Unknown"
    }


@lru_cache(maxsize=500)
def fetch_school_data(postcode):

    result = propertydata_request(
        "schools",
        postcode=postcode
    )

    data = result.get("data", {})

    state_schools = (
        data
        .get("state", {})
        .get("nearest", [])
    )

    independent_schools = (
        data
        .get("independent", {})
        .get("nearest", [])
    )

    schools = state_schools + independent_schools

    if not schools:
        raise RuntimeError(
            f"No schools returned for {postcode}. "
            f"Response: {result}"
        )

    valid = []

    for school in schools:

        try:
            distance = float(
                school.get("distance")
            )
        except (TypeError, ValueError):
            continue

        valid.append(
            (distance, school)
        )

    if not valid:
        raise RuntimeError(
            f"No usable school distances for {postcode}"
        )

    distance, nearest = min(
        valid,
        key=lambda x: x[0]
    )

    return {
        "school_nearby": distance <= 1.0,
        "school_distance": distance,
        "nearest_school": nearest.get(
            "name",
            "Unknown"
        )
    }


# =========================================================
# HOSPITAL / STATION PLACEHOLDERS
# =========================================================
#
# Replace these later with a relevant location API.
#
# True   = nearby
# False  = not nearby
# None   = we do not yet know
# =========================================================

def fetch_hospital_data(postcode):

    return {
        "hospital_nearby": None,
        "hospital_distance": None
    }


def fetch_station_data(postcode):

    return {
        "station_nearby": None,
        "station_distance": None
    }


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_growth_from_row(row):

    input_data = {
        feature: row[feature]
        for feature in model_features
    }

    X_new = pd.DataFrame(
        [input_data],
        columns=model_features
    )

    prediction = growth_model.predict(
        X_new
    )[0]

    return float(prediction)


def add_model_predictions(candidate_df):

    candidate_df = (
        candidate_df.copy()
    )

    candidate_df[
        "predicted_growth"
    ] = candidate_df.apply(
        predict_growth_from_row,
        axis=1
    )

    candidate_df[
        "predicted_future_price"
    ] = (
        candidate_df["price_data_7"]
        *
        (
            1
            +
            candidate_df[
                "predicted_growth"
            ]
            / 100
        )
    )

    return candidate_df


# =========================================================
# INITIAL SHORTLIST
# =========================================================

def create_initial_shortlist(
    budget_value,
    max_candidates=8
):

    candidates = df.copy()

    # Remove invalid rows first
    candidates = candidates[
        candidates["price_data_7"] > 0
    ].copy()

    # Add model predictions
    candidates = add_model_predictions(
        candidates
    )

    # Allow up to 20% above budget
    budget_limit = (
        budget_value * 1.20
    )

    candidates = candidates[
        candidates["price_data_7"]
        <= budget_limit
    ]

    # Initial model shortlist
    candidates = (
        candidates
        .sort_values(
            "predicted_growth",
            ascending=False
        )
        .head(max_candidates)
    )

    return candidates


# =========================================================
# LIVE API ENRICHMENT
# =========================================================

def enrich_candidate(
    row,
    bedrooms_value
):

    postcode = str(
        row["postcode"]
    ).strip().upper()

    output = row.to_dict()

    output["postcode"] = postcode


    # -------------------------
    # LIVE PRICE
    # -------------------------

    try:

        output["live_price"] = (
            fetch_live_price(
                postcode,
                bedrooms_value
            )
        )

    except Exception as e:

        print(
            f"{postcode}: "
            f"price unavailable ({e})"
        )

        output["live_price"] = float(
            row["price_data_7"]
        )


    # -------------------------
    # CRIME
    # -------------------------

    try:

        crime = fetch_crime(
            postcode
        )
    
        output.update(
            crime
        )
    
    except Exception as e:
    
        print(
            f"[CRIME ERROR] "
            f"{postcode}: {e}"
        )
    
        output["crime_rate"] = np.nan
        output["crime_rating"] = "Unavailable"


    # -------------------------
    # SCHOOL
    # -------------------------

    try:

        school = fetch_school_data(
            postcode
        )
    
        output.update(
            school
        )
    
    except Exception as e:
    
        print(
            f"[SCHOOL ERROR] "
            f"{postcode}: {e}"
        )
    
        output["school_nearby"] = None
        output["school_distance"] = None
        output["nearest_school"] = None


    # -------------------------
    # HOSPITAL
    # -------------------------

    try:

        output.update(
            fetch_hospital_data(
                postcode
            )
        )

    except Exception:

        output.update({
            "hospital_nearby": None,
            "hospital_distance": None
        })


    # -------------------------
    # TRAIN STATION
    # -------------------------

    try:

        output.update(
            fetch_station_data(
                postcode
            )
        )

    except Exception:

        output.update({
            "station_nearby": None,
            "station_distance": None
        })


    # -------------------------
    # FUTURE PRICE
    # -------------------------

    output[
        "predicted_future_price"
    ] = (
        output["live_price"]
        *
        (
            1
            +
            output["predicted_growth"]
            / 100
        )
    )

    return output


# =========================================================
# PARALLEL API FETCHING
# =========================================================

def enrich_shortlist_parallel(
    shortlist,
    bedrooms_value,
    max_workers=4
):

    enriched = []

    rows = [
        row
        for _, row
        in shortlist.iterrows()
    ]

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                enrich_candidate,
                row,
                bedrooms_value
            ): row["postcode"]

            for row in rows
        }

        for future in as_completed(
            futures
        ):

            postcode = futures[
                future
            ]

            try:

                result = (
                    future.result()
                )

                enriched.append(
                    result
                )

            except Exception as e:

                print(
                    f"Could not load "
                    f"{postcode}: {e}"
                )

    return enriched


# =========================================================
# NORMALISATION
# =========================================================

def normalise(
    values,
    higher_is_better=True
):

    values = np.array(
        values,
        dtype=float
    )

    valid = values[
        ~np.isnan(values)
    ]

    if len(valid) == 0:
        return np.ones(
            len(values)
        )

    low = np.nanmin(values)
    high = np.nanmax(values)

    if high == low:

        return np.ones(
            len(values)
        )

    if higher_is_better:

        return (
            values - low
        ) / (
            high - low
        )

    return (
        high - values
    ) / (
        high - low
    )


# =========================================================
# AMENITY DISPLAY
# =========================================================

def amenity_icon(value):

    if value is True:
        return "✅"

    if value is False:
        return "❌"

    return "—"


# =========================================================
# BASIC INPUTS
# =========================================================

budget = widgets.IntText(
    value=400000,
    description="Budget (£):",
    style={
        "description_width": "100px"
    },
    layout=widgets.Layout(
        width="100%"
    )
)


bedrooms = widgets.Dropdown(
    options=[
        ("Any", 0),
        ("1 bedroom", 1),
        ("2 bedrooms", 2),
        ("3 bedrooms", 3),
        ("4 bedrooms", 4),
        ("5+ bedrooms", 5)
    ],

    value=0,

    description="Bedrooms:",

    style={
        "description_width": "100px"
    },

    layout=widgets.Layout(
        width="100%"
    )
)


# =========================================================
# AMENITIES
# =========================================================

school_checkbox = widgets.Checkbox(
    value=False,
    description="School nearby",
    indent=False
)

hospital_checkbox = widgets.Checkbox(
    value=False,
    description="Hospital nearby",
    indent=False
)

station_checkbox = widgets.Checkbox(
    value=False,
    description="Train station nearby",
    indent=False
)


amenity_box = widgets.VBox(
    [
        school_checkbox,
        hospital_checkbox,
        station_checkbox
    ],

    layout=widgets.Layout(
        gap="4px"
    )
)


# =========================================================
# WEIGHTS + LOCKS
# =========================================================

names = {
    "price":
        "Sale Price",

    "growth":
        "Predicted Growth",

    "crime":
        "Crime & Safety"
}

defaults = {
    "price": 40,
    "growth": 40,
    "crime": 20
}

keys = list(
    names.keys()
)

sliders = {}
numbers = {}
lock_buttons = {}

weight_rows = []

busy = False


# =========================================================
# OUTPUTS
# =========================================================

total_label = widgets.HTML()

chart_output = widgets.Output()

results_output = widgets.Output()

comparison_output = widgets.Output()


# =========================================================
# LOCK HANDLER
# =========================================================

def lock_changed(
    change,
    key
):

    if change[
        "name"
    ] != "value":
        return

    locked = change[
        "new"
    ]

    lock_buttons[
        key
    ].description = (
        "🔒"
        if locked
        else "🔓"
    )

    lock_buttons[
        key
    ].tooltip = (
        "Unlock this weight"
        if locked
        else "Lock this weight"
    )


# =========================================================
# BUILD SLIDERS
# =========================================================

for key in keys:

    label = widgets.HTML(
        value=(
            f"<b>{names[key]}</b>"
        ),
        layout=widgets.Layout(
            width="125px",
            min_width="125px"
        )
    )


    slider = widgets.IntSlider(
        value=defaults[key],
        min=0,
        max=100,
        step=1,
        readout=False,
        continuous_update=True,

        layout=widgets.Layout(
            width="100%",
            min_width="110px",
            flex="1 1 auto"
        )
    )


    number = widgets.BoundedIntText(
        value=defaults[key],
        min=0,
        max=100,
        step=1,

        layout=widgets.Layout(
            width="65px",
            min_width="65px"
        )
    )


    percent = widgets.HTML(
        "%",
        layout=widgets.Layout(
            width="15px"
        )
    )


    lock_button = widgets.ToggleButton(
        value=False,
        description="🔓",
        tooltip="Lock this weight",

        layout=widgets.Layout(
            width="48px",
            min_width="48px"
        )
    )


    sliders[key] = slider
    numbers[key] = number
    lock_buttons[key] = (
        lock_button
    )


    row = widgets.HBox(
        [
            label,
            slider,
            number,
            percent,
            lock_button
        ],

        layout=widgets.Layout(
            width="100%",
            align_items="center",
            gap="5px"
        )
    )


    weight_rows.append(
        row
    )


preferences_box = widgets.VBox(
    weight_rows,
    layout=widgets.Layout(
        width="100%",
        gap="6px"
    )
)


# NOW attach lock listeners.
# This must happen AFTER the buttons exist.
for key in keys:

    lock_buttons[
        key
    ].observe(

        lambda change, k=key:
        lock_changed(
            change,
            k
        ),

        names="value"
    )


# =========================================================
# CHART + TOTAL
# =========================================================

def refresh_total():

    total = sum(
        numbers[k].value
        for k in keys
    )

    colour = (
        "#16803c"
        if total == 100
        else "#b42318"
    )

    total_label.value = (
        f"<b style='color:{colour}'>"
        f"Total preference: "
        f"{total}% / 100%"
        f"</b>"
    )


def refresh_chart():

    values = [
        numbers[k].value
        for k in keys
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[
                    names[k]
                    for k in keys
                ],

                values=values,

                hole=0.58,

                sort=False,

                textinfo="percent",

                hovertemplate=(
                    "%{label}: "
                    "%{value}%"
                    "<extra></extra>"
                )
            )
        ]
    )


    fig.update_layout(
        title=(
            "Preference distribution"
        ),

        height=285,

        margin=dict(
            l=10,
            r=10,
            t=45,
            b=15
        ),

        showlegend=True,

        legend=dict(
            orientation="h",
            y=-0.03,
            x=0.5,
            xanchor="center"
        )
    )


    with chart_output:

        clear_output(
            wait=True
        )

        display(fig)


# =========================================================
# WEIGHT REDISTRIBUTION WITH LOCK SUPPORT
# =========================================================

def change_weight(
    changed_key,
    new_value
):

    global busy

    if busy:
        return


    # If the slider is locked,
    # restore its current stored value.
    if lock_buttons[
        changed_key
    ].value:

        busy = True

        try:

            sliders[
                changed_key
            ].value = numbers[
                changed_key
            ].value

        finally:

            busy = False

        return


    new_value = int(
        max(
            0,
            min(
                100,
                new_value
            )
        )
    )


    locked_keys = [
        k
        for k in keys
        if lock_buttons[k].value
    ]


    unlocked_keys = [
        k
        for k in keys
        if not lock_buttons[k].value
    ]


    locked_total = sum(
        numbers[k].value
        for k in locked_keys
    )


    available_total = (
        100 - locked_total
    )


    # Only one unlocked slider:
    # it must equal all remaining weight.
    if len(
        unlocked_keys
    ) == 1:

        only_key = (
            unlocked_keys[0]
        )

        busy = True

        try:

            sliders[
                only_key
            ].value = available_total

            numbers[
                only_key
            ].value = available_total

        finally:

            busy = False


        refresh_total()
        refresh_chart()

        return


    new_value = min(
        new_value,
        available_total
    )


    other_unlocked = [
        k
        for k in unlocked_keys
        if k != changed_key
    ]


    remaining = (
        available_total
        - new_value
    )


    old_values = {
        k:
        max(
            0,
            numbers[k].value
        )

        for k
        in other_unlocked
    }


    old_total = sum(
        old_values.values()
    )


    if old_total == 0:

        exact = {
            k:
            remaining /
            len(
                other_unlocked
            )

            for k
            in other_unlocked
        }

    else:

        exact = {
            k:
            remaining
            * old_values[k]
            / old_total

            for k
            in other_unlocked
        }


    allocated = {
        k:
        int(exact[k])

        for k
        in other_unlocked
    }


    leftover = (
        remaining
        -
        sum(
            allocated.values()
        )
    )


    order = sorted(
        other_unlocked,

        key=lambda k:
        exact[k]
        -
        allocated[k],

        reverse=True
    )


    for k in order[
        :leftover
    ]:

        allocated[k] += 1


    busy = True

    try:

        numbers[
            changed_key
        ].value = new_value

        sliders[
            changed_key
        ].value = new_value


        for k in other_unlocked:

            numbers[
                k
            ].value = (
                allocated[k]
            )

            sliders[
                k
            ].value = (
                allocated[k]
            )

    finally:

        busy = False


    refresh_total()
    refresh_chart()


# =========================================================
# SLIDER CALLBACKS
# =========================================================

def on_slider(
    change,
    key
):

    if (
        change["name"]
        == "value"
        and not busy
    ):

        change_weight(
            key,
            change["new"]
        )


def on_number(
    change,
    key
):

    if (
        change["name"]
        == "value"
        and not busy
    ):

        change_weight(
            key,
            change["new"]
        )


for key in keys:

    sliders[
        key
    ].observe(

        lambda change, k=key:
        on_slider(
            change,
            k
        ),

        names="value"
    )


    numbers[
        key
    ].observe(

        lambda change, k=key:
        on_number(
            change,
            k
        ),

        names="value"
    )


# =========================================================
# SCORE CANDIDATES
# =========================================================

def score_candidates(
    candidates
):

    if not candidates:
        return []


    prices = [
        float(
            c["live_price"]
        )
        for c in candidates
    ]


    growth_values = [
        float(
            c["predicted_growth"]
        )
        for c in candidates
    ]


    valid_crimes = [
        float(
            c["crime_rate"]
        )
        for c in candidates
        if not pd.isna(
            c.get(
                "crime_rate",
                np.nan
            )
        )
    ]


    crime_fallback = (
        np.median(valid_crimes)
        if valid_crimes
        else 0
    )


    crime_values = [

        float(
            c["crime_rate"]
        )

        if not pd.isna(
            c.get(
                "crime_rate",
                np.nan
            )
        )

        else crime_fallback

        for c in candidates
    ]


    price_score = normalise(
        prices,
        higher_is_better=False
    )


    growth_score = normalise(
        growth_values,
        higher_is_better=True
    )


    crime_score = normalise(
        crime_values,
        higher_is_better=False
    )


    weights = {
        key:
        numbers[key].value
        / 100

        for key in keys
    }


    output = []


    for i, candidate in enumerate(
        candidates
    ):

        # -------------------------------------
        # USER'S AMENITY REQUIREMENTS
        # -------------------------------------

        if school_checkbox.value:

            if candidate.get(
                "school_nearby"
            ) is False:

                continue


        # Only filter hospital/station when
        # actual True/False data becomes available.
        #
        # None currently means "unknown".

        if hospital_checkbox.value:

            if candidate.get(
                "hospital_nearby"
            ) is False:

                continue


        if station_checkbox.value:

            if candidate.get(
                "station_nearby"
            ) is False:

                continue


        score = 100 * (

            weights["price"]
            * price_score[i]

            +

            weights["growth"]
            * growth_score[i]

            +

            weights["crime"]
            * crime_score[i]

        )


        result = (
            candidate.copy()
        )

        result["score"] = (
            float(score)
        )

        output.append(
            result
        )


    return sorted(
        output,

        key=lambda x:
        x["score"],

        reverse=True
    )


# =========================================================
# COMPARISON CHECKBOXES
# =========================================================

current_recommendations = []

comparison_checkboxes = []

comparison_selection_box = (
    widgets.VBox()
)


def update_comparison_options():

    global comparison_checkboxes

    comparison_checkboxes = []

    items = []


    for p in current_recommendations:

        checkbox = widgets.Checkbox(
            value=False,
            description=p["postcode"],
            indent=False
        )

        comparison_checkboxes.append(
            (
                checkbox,
                p["postcode"]
            )
        )

        items.append(
            checkbox
        )


    comparison_selection_box.children = tuple(
        items
    )


# =========================================================
# RECOMMEND
# =========================================================

def recommend(
    _=None
):

    global current_recommendations


    with results_output:

        clear_output(
            wait=True
        )

        display(
            widgets.HTML(
                """
                <div style="
                    padding:15px;
                    border:1px solid #ddd;
                    border-radius:8px;
                ">
                    <b>
                        Generating model shortlist...
                    </b>
                </div>
                """
            )
        )


    # -----------------------------------------
    # MODEL SHORTLIST
    # -----------------------------------------

    shortlist = (
        create_initial_shortlist(
            budget.value,
            max_candidates=8
        )
    )


    if shortlist.empty:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    """
                    <b style="color:#b42318">
                        No areas found for this budget.
                    </b>
                    """
                )
            )

        return


    with results_output:

        clear_output(
            wait=True
        )

        display(
            widgets.HTML(
                f"""
                <div style="
                    padding:15px;
                    border:1px solid #ddd;
                    border-radius:8px;
                ">
                    <b>
                        Fetching live data for
                        {len(shortlist)}
                        candidate areas...
                    </b>
                </div>
                """
            )
        )


    # -----------------------------------------
    # PARALLEL API ENRICHMENT
    # -----------------------------------------

    enriched = (
        enrich_shortlist_parallel(
            shortlist,
            bedrooms.value,
            max_workers=4
        )
    )


    if not enriched:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    """
                    <b style="color:#b42318">
                        No areas could be loaded.
                    </b>
                    """
                )
            )

        return


    # -----------------------------------------
    # FINAL SCORE
    # -----------------------------------------

    ranked = score_candidates(
        enriched
    )


    current_recommendations = (
        ranked[:5]
    )


    update_comparison_options()


    if not current_recommendations:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    """
                    <b style="color:#b42318">
                        No areas matched the selected
                        requirements.
                    </b>
                    """
                )
            )

        return


    cards = []


    for rank, p in enumerate(
        current_recommendations,
        start=1
    ):

        live_price = float(
            p.get(
                "live_price",
                0
            )
        )

        predicted_growth = float(
            p.get(
                "predicted_growth",
                0
            )
        )

        future_price = float(
            p.get(
                "predicted_future_price",
                0
            )
        )

        score = float(
            p.get(
                "score",
                0
            )
        )


        school_icon = amenity_icon(
            p.get(
                "school_nearby"
            )
        )


        hospital_icon = amenity_icon(
            p.get(
                "hospital_nearby"
            )
        )


        station_icon = amenity_icon(
            p.get(
                "station_nearby"
            )
        )


        nearest_school = (
            p.get(
                "nearest_school"
            )
            or "Unavailable"
        )


        school_distance = (
            p.get(
                "school_distance"
            )
        )


        if (
            school_distance
            is not None
        ):

            school_detail = (
                f"{nearest_school} "
                f"({float(school_distance):.1f} mi)"
            )

        else:

            school_detail = (
                "Unavailable"
            )


        cards.append(
            f"""
            <div style="
                border:1px solid #d9d9d9;
                border-radius:12px;
                padding:18px;
                margin:0 0 14px 0;
                background:white;
                color:#222;
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    gap:15px;
                ">

                    <div>

                        <div style="
                            color:#777;
                            font-size:12px;
                            font-weight:bold;
                        ">
                            RANK #{rank}
                        </div>

                        <div style="
                            font-size:23px;
                            font-weight:bold;
                        ">
                            {p['postcode']}
                        </div>

                    </div>

                    <div style="
                        font-size:25px;
                        font-weight:bold;
                    ">
                        {score:.0f}/100
                    </div>

                </div>

                <hr>

                <div style="
                    display:grid;
                    grid-template-columns:
                        repeat(
                            auto-fit,
                            minmax(140px, 1fr)
                        );
                    gap:15px;
                ">

                    <div>
                        <span style="color:#666">
                            Current asking price
                        </span>
                        <br>
                        <b>
                            £{live_price:,.0f}
                        </b>
                    </div>

                    <div>
                        <span style="color:#666">
                            Predicted growth
                        </span>
                        <br>
                        <b>
                            {predicted_growth:.2f}%
                        </b>
                    </div>

                    <div>
                        <span style="color:#666">
                            Predicted future price
                        </span>
                        <br>
                        <b>
                            £{future_price:,.0f}
                        </b>
                    </div>

                    <div>
                        <span style="color:#666">
                            Crime
                        </span>
                        <br>
                        <b>
                            {p.get(
                                'crime_rating',
                                'Unknown'
                            )}
                        </b>
                    </div>

                </div>

                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:20px;
                    margin-top:18px;
                    padding-top:12px;
                    border-top:1px solid #eee;
                ">

                    <div>
                        {school_icon}
                        <b> School</b>
                    </div>

                    <div>
                        {hospital_icon}
                        <b> Hospital</b>
                    </div>

                    <div>
                        {station_icon}
                        <b> Train station</b>
                    </div>

                </div>

                <div style="
                    margin-top:9px;
                    color:#777;
                    font-size:12px;
                ">
                    Nearest school:
                    {school_detail}
                </div>

            </div>
            """
        )


    with results_output:

        clear_output(
            wait=True
        )

        display(
            widgets.HTML(
                "".join(cards)
            )
        )


# =========================================================
# COMPARE 2–5 AREAS
# =========================================================

def compare_selected(
    _=None
):

    with comparison_output:

        clear_output(
            wait=True
        )


        selected_codes = [

            postcode

            for checkbox, postcode
            in comparison_checkboxes

            if checkbox.value
        ]


        if len(
            selected_codes
        ) < 2:

            display(
                widgets.HTML(
                    """
                    <b style="color:#b42318">
                        Please select at least
                        two areas to compare.
                    </b>
                    """
                )
            )

            return


        selected = [

            p

            for p
            in current_recommendations

            if p["postcode"]
            in selected_codes
        ]


        rows = ""


        for p in selected:

            live_price = float(
                p.get(
                    "live_price",
                    0
                )
            )


            predicted_growth = float(
                p.get(
                    "predicted_growth",
                    0
                )
            )


            future_price = float(
                p.get(
                    "predicted_future_price",
                    0
                )
            )


            score = float(
                p.get(
                    "score",
                    0
                )
            )


            crime = p.get(
                "crime_rating",
                "Unknown"
            )


            school_icon = amenity_icon(
                p.get(
                    "school_nearby"
                )
            )


            hospital_icon = amenity_icon(
                p.get(
                    "hospital_nearby"
                )
            )


            station_icon = amenity_icon(
                p.get(
                    "station_nearby"
                )
            )


            rows += f"""
            <tr style="
                border-bottom:1px solid #ddd;
            ">

                <td style="padding:10px">
                    <b>
                        {p['postcode']}
                    </b>
                </td>

                <td style="padding:10px">
                    £{live_price:,.0f}
                </td>

                <td style="padding:10px">
                    {predicted_growth:.2f}%
                </td>

                <td style="padding:10px">
                    £{future_price:,.0f}
                </td>

                <td style="padding:10px">
                    {crime}
                </td>

                <td style="
                    padding:10px;
                    text-align:center;
                    font-size:20px;
                ">
                    {school_icon}
                </td>

                <td style="
                    padding:10px;
                    text-align:center;
                    font-size:20px;
                ">
                    {hospital_icon}
                </td>

                <td style="
                    padding:10px;
                    text-align:center;
                    font-size:20px;
                ">
                    {station_icon}
                </td>

                <td style="padding:10px">
                    <b>
                        {score:.0f}/100
                    </b>
                </td>

            </tr>
            """


        display(
            widgets.HTML(
                f"""
                <h3>
                    Comparing
                    {len(selected)}
                    areas
                </h3>

                <div style="
                    width:100%;
                    overflow-x:auto;
                ">

                    <table style="
                        width:100%;
                        min-width:850px;
                        border-collapse:collapse;
                        font-size:13px;
                    ">

                        <thead>

                            <tr style="
                                background:#f5f5f5;
                                color:#222;
                            ">

                                <th style="padding:10px">
                                    Area
                                </th>

                                <th style="padding:10px">
                                    Price
                                </th>

                                <th style="padding:10px">
                                    Growth
                                </th>

                                <th style="padding:10px">
                                    Future Price
                                </th>

                                <th style="padding:10px">
                                    Crime
                                </th>

                                <th style="padding:10px">
                                    School
                                </th>

                                <th style="padding:10px">
                                    Hospital
                                </th>

                                <th style="padding:10px">
                                    Station
                                </th>

                                <th style="padding:10px">
                                    Score
                                </th>

                            </tr>

                        </thead>

                        <tbody>
                            {rows}
                        </tbody>

                    </table>

                </div>

                <p style="
                    color:#777;
                    font-size:11px;
                    margin-top:10px;
                ">
                    ✅ nearby &nbsp;&nbsp;
                    ❌ not nearby &nbsp;&nbsp;
                    — data currently unavailable
                </p>
                """
            )
        )


# =========================================================
# BUTTONS
# =========================================================

recommend_button = widgets.Button(
    description="Find best areas",
    button_style="primary",
    icon="search",

    layout=widgets.Layout(
        width="100%",
        height="44px"
    )
)


recommend_button.on_click(
    recommend
)


compare_button = widgets.Button(
    description="Compare selected areas",
    button_style="info",
    icon="exchange",

    layout=widgets.Layout(
        width="240px",
        height="40px"
    )
)


compare_button.on_click(
    compare_selected
)


# =========================================================
# FINAL VERTICAL LAYOUT
# =========================================================

app = widgets.VBox(
    [

        widgets.HTML(
            """
            <div>
                <h2 style="
                    margin:0 0 3px 0;
                ">
                    HomeMatch AI
                </h2>

                <p style="
                    margin:0;
                    color:#777;
                ">
                    AI-powered UK property
                    area recommendation
                </p>
            </div>
            """
        ),


        widgets.HTML(
            """
            <hr>
            <h3>
                Property requirements
            </h3>
            """
        ),

        widgets.HBox(
            [
                budget,
                bedrooms
            ],

            layout=widgets.Layout(
                gap="20px",
                flex_flow="row wrap"
            )
        ),


        widgets.HTML(
            """
            <h3>
                Nearby amenities
            </h3>
            """
        ),

        amenity_box,


        widgets.HTML(
            """
            <small style="color:#777">
                Tick amenities you would like nearby.
                A dash in the results means that
                data source has not yet been connected.
            </small>
            """
        ),


        widgets.HTML(
            """
            <h3>
                Your priorities
            </h3>
            """
        ),

        preferences_box,

        total_label,


        widgets.HTML(
            """
            <small style="color:#777">
                🔒 Lock a preference so changing
                another slider does not alter it.
            </small>
            """
        ),


        widgets.HTML(
            """
            <h3>
                Preference distribution
            </h3>
            """
        ),

        chart_output,


        recommend_button,


        widgets.HTML(
            """
            <hr>

            <h2>
                Recommended areas
            </h2>

            <p style="color:#777">
                Your top matches based on
                budget, priorities and
                predicted price growth.
            </p>
            """
        ),

        results_output,


        widgets.HTML(
            """
            <hr>

            <h2>
                Compare recommendations
            </h2>

            <p style="color:#777">
                Tick any two or more
                recommended areas below.
            </p>
            """
        ),

        comparison_selection_box,

        compare_button,

        comparison_output

    ],

    layout=widgets.Layout(
        width="100%",
        max_width="1100px",
        padding="12px",
        gap="10px"
    )
)


# =========================================================
# INITIAL DISPLAY
# =========================================================

refresh_total()
refresh_chart()

display(app)