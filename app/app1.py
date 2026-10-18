# =========================================================
# PropertyFinder
# Buy / Rent / Invest
# + Average Bedrooms
# =========================================================

import os
import re
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

from dotenv import load_dotenv
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "../data/property_price_data.csv"
MODEL_PATH = "../models/future_growth_rf_v1.pkl"

PROPERTYDATA_BASE_URL = "https://api.propertydata.co.uk"

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print("Looking for .env at:", ENV_PATH)
print(".env exists:", ENV_PATH.exists())

# Load variables from .env
load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)

PROPERTYDATA_API_KEY = os.getenv(
    "PROPERTYDATA_API_KEY"
)

print(
    "API key loaded:",
    bool(PROPERTYDATA_API_KEY)
)

GOOD_INTERNET_MBPS = 50
RESTAURANT_DISTANCE_MILES = 1.0

MODEL_FEATURES = [
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
        "WARNING: PROPERTYDATA_API_KEY is not set.\n\n"
        "Run this in a cell BEFORE this app:\n\n"
        'import os\n'
        'os.environ["PROPERTYDATA_API_KEY"] = "YOUR_KEY"'
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

        if response.status_code == 429:

            wait_seconds = int(
                response.headers.get(
                    "Retry-After",
                    5
                )
            )

            print(
                f"Rate limit reached. "
                f"Waiting {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

            continue

        try:

            data = response.json()

        except Exception:

            raise RuntimeError(
                f"Invalid response from "
                f"{endpoint}: "
                f"{response.text[:300]}"
            )

        if response.ok:

            if data.get("status") == "success":

                return data

            raise RuntimeError(
                f"{data.get('code')} - "
                f"{data.get('message')}"
            )

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{data}"
        )

    raise RuntimeError(
        "PropertyData request repeatedly throttled."
    )


# =========================================================
# POSTCODE HELPERS
# =========================================================

FULL_POSTCODE_RE = re.compile(
    r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$",
    re.IGNORECASE
)


def clean_postcode(postcode):

    return (
        str(postcode)
        .strip()
        .upper()
    )


def is_full_postcode(postcode):

    return bool(
        FULL_POSTCODE_RE.match(
            clean_postcode(
                postcode
            )
        )
    )


def format_full_postcode(postcode):

    value = (
        clean_postcode(
            postcode
        )
        .replace(
            " ",
            ""
        )
    )

    if len(value) <= 3:

        return value

    return (
        value[:-3]
        + " "
        + value[-3:]
    )


def _extract_full_postcodes(obj):

    found = []

    if isinstance(
        obj,
        dict
    ):

        for key, value in obj.items():

            if (
                str(key).lower() == "postcode"
                and isinstance(
                    value,
                    str
                )
                and is_full_postcode(
                    value
                )
            ):

                found.append(
                    format_full_postcode(
                        value
                    )
                )

            found.extend(
                _extract_full_postcodes(
                    value
                )
            )

    elif isinstance(
        obj,
        list
    ):

        for item in obj:

            found.extend(
                _extract_full_postcodes(
                    item
                )
            )

    return found


@lru_cache(maxsize=500)
def resolve_full_postcode(
    area_postcode
):

    area_postcode = clean_postcode(
        area_postcode
    )

    if is_full_postcode(
        area_postcode
    ):

        return format_full_postcode(
            area_postcode
        )

    try:

        result = propertydata_request(
            "uprns",
            postcode=area_postcode,
            results=10,
            strict="true"
        )

        possible = list(
            dict.fromkeys(
                _extract_full_postcodes(
                    result
                )
            )
        )

        if possible:

            return possible[0]

    except Exception as e:

        print(
            f"[POSTCODE ERROR] "
            f"{area_postcode}: "
            f"{e}"
        )

    return None


# =========================================================
# LIVE PURCHASE PRICE
# =========================================================

@lru_cache(maxsize=500)
def fetch_live_price(
    postcode,
    bedrooms=0
):

    params = {
        "postcode":
            postcode
    }

    if bedrooms:

        params[
            "bedrooms"
        ] = bedrooms

    result = propertydata_request(
        "prices",
        **params
    )

    average = (
        result
        .get(
            "data",
            {}
        )
        .get(
            "average"
        )
    )

    if average is None:

        raise RuntimeError(
            f"No asking price returned "
            f"for {postcode}"
        )

    return float(
        average
    )


# =========================================================
# LIVE WEEKLY RENT
# =========================================================

@lru_cache(maxsize=500)
def fetch_live_rent(
    postcode,
    bedrooms=0
):

    params = {
        "postcode":
            postcode
    }

    if bedrooms:

        params[
            "bedrooms"
        ] = bedrooms

    result = propertydata_request(
        "rents",
        **params
    )

    # IMPORTANT:
    # rents are nested under data -> long_let
    long_let = (
        result
        .get(
            "data",
            {}
        )
        .get(
            "long_let",
            {}
        )
    )

    average = long_let.get(
        "average"
    )

    if average is None:

        raise RuntimeError(
            f"No weekly rent returned "
            f"for {postcode}"
        )

    return float(
        average
    )


# =========================================================
# AVERAGE BEDROOMS
# =========================================================

@lru_cache(maxsize=500)
def fetch_average_bedrooms(
    postcode,
    mode
):

    # -----------------------------------------
    # RENT LISTINGS
    # -----------------------------------------

    if mode == "rent":

        result = propertydata_request(
            "rents",
            postcode=postcode,
            points=50
        )

        raw_data = (
            result
            .get(
                "data",
                {}
            )
            .get(
                "long_let",
                {}
            )
            .get(
                "raw_data",
                []
            )
        )

    # -----------------------------------------
    # BUY / INVEST LISTINGS
    # -----------------------------------------

    else:

        result = propertydata_request(
            "prices",
            postcode=postcode,
            points=50
        )

        raw_data = (
            result
            .get(
                "data",
                {}
            )
            .get(
                "raw_data",
                []
            )
        )


    bedroom_values = []


    for property_item in raw_data:

        bedroom_value = (
            property_item
            .get(
                "bedrooms"
            )
        )

        if bedroom_value is None:

            continue

        try:

            bedroom_value = float(
                bedroom_value
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        if (
            bedroom_value >= 0
            and bedroom_value <= 10
        ):

            bedroom_values.append(
                bedroom_value
            )


    if not bedroom_values:

        return None


    return float(
        np.mean(
            bedroom_values
        )
    )


# =========================================================
# ROUND BEDROOMS
# =========================================================

def display_bedrooms(
    average_bedrooms
):

    if average_bedrooms is None:

        return "—"

    try:

        # Conventional rounding:
        # 2.4 -> 2
        # 2.5 -> 3
        # 2.8 -> 3
        rounded = int(
            np.floor(
                float(
                    average_bedrooms
                )
                + 0.5
            )
        )

        return str(
            rounded
        )

    except Exception:

        return "—"


# =========================================================
# CRIME
# =========================================================

@lru_cache(maxsize=500)
def fetch_crime(postcode):

    result = propertydata_request(
        "crime",
        postcode=postcode
    )

    crime_rate = result.get(
        "crimes_per_thousand"
    )

    crime_rating = result.get(
        "crime_rating"
    )

    if crime_rate is None:

        raise RuntimeError(
            f"No crime rate returned "
            f"for {postcode}"
        )

    return {
        "crime_rate":
            float(
                crime_rate
            ),

        "crime_rating":
            crime_rating
            or "Unknown"
    }


# =========================================================
# SCHOOLS
# =========================================================

@lru_cache(maxsize=500)
def fetch_school_data(postcode):

    result = propertydata_request(
        "schools",
        postcode=postcode
    )

    data = result.get(
        "data",
        {}
    )

    schools = []

    schools.extend(
        data
        .get(
            "state",
            {}
        )
        .get(
            "nearest",
            []
        )
    )

    schools.extend(
        data
        .get(
            "independent",
            {}
        )
        .get(
            "nearest",
            []
        )
    )

    valid = []


    for school in schools:

        try:

            distance = float(
                school.get(
                    "distance"
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        valid.append(
            (
                distance,
                school
            )
        )


    if not valid:

        raise RuntimeError(
            f"No usable school data "
            f"for {postcode}"
        )


    distance, nearest = min(
        valid,
        key=lambda x:
            x[0]
    )


    return {
        "school_nearby":
            distance <= 1.0,

        "school_distance":
            distance,

        "nearest_school":
            nearest.get(
                "name",
                "Unknown"
            )
    }


# =========================================================
# GENERIC JSON WALKER
# =========================================================

def _walk_dicts(obj):

    found = []

    if isinstance(
        obj,
        dict
    ):

        found.append(
            obj
        )

        for value in obj.values():

            found.extend(
                _walk_dicts(
                    value
                )
            )

    elif isinstance(
        obj,
        list
    ):

        for item in obj:

            found.extend(
                _walk_dicts(
                    item
                )
            )

    return found


# =========================================================
# RESTAURANTS
# =========================================================

@lru_cache(maxsize=500)
def fetch_restaurant_data(
    postcode
):

    result = propertydata_request(
        "restaurants",
        postcode=postcode
    )

    restaurant_records = []


    for record in _walk_dicts(
        result
    ):

        distance = record.get(
            "distance"
        )

        name = (
            record.get(
                "name"
            )
            or record.get(
                "restaurant_name"
            )
            or record.get(
                "business_name"
            )
        )


        if (
            distance is not None
            and name is not None
        ):

            try:

                restaurant_records.append({
                    "name":
                        str(
                            name
                        ),

                    "distance":
                        float(
                            distance
                        )
                })

            except (
                TypeError,
                ValueError
            ):

                pass


    unique = {}


    for restaurant in restaurant_records:

        key = (
            restaurant[
                "name"
            ],

            restaurant[
                "distance"
            ]
        )

        unique[
            key
        ] = restaurant


    restaurant_records = list(
        unique.values()
    )


    if not restaurant_records:

        return {
            "restaurants_nearby":
                None,

            "restaurant_count":
                None,

            "nearest_restaurant_distance":
                None
        }


    nearest = min(
        r[
            "distance"
        ]

        for r in restaurant_records
    )


    return {
        "restaurants_nearby":
            nearest
            <= RESTAURANT_DISTANCE_MILES,

        "restaurant_count":
            len(
                restaurant_records
            ),

        "nearest_restaurant_distance":
            nearest
    }


# =========================================================
# FLOOD RISK
# =========================================================

@lru_cache(maxsize=500)
def fetch_flood_data(
    postcode
):

    result = propertydata_request(
        "flood-risk",
        postcode=postcode
    )

    flood_risk = result.get(
        "flood_risk"
    )

    if flood_risk is None:

        raise RuntimeError(
            f"No flood risk returned "
            f"for {postcode}"
        )


    flood_risk = str(
        flood_risk
    ).strip()


    low_flood_risk = (
        flood_risk.lower()
        in {
            "low",
            "very low"
        }
    )


    return {
        "flood_risk":
            flood_risk,

        "low_flood_risk":
            low_flood_risk
    }


# =========================================================
# INTERNET
# =========================================================

@lru_cache(maxsize=500)
def fetch_internet_data(
    postcode
):

    result = propertydata_request(
        "internet-speed",
        postcode=postcode
    )

    candidates = []


    def extract_number(
        value
    ):

        if isinstance(
            value,
            (int, float)
        ):

            return float(
                value
            )


        if isinstance(
            value,
            str
        ):

            match = re.search(
                r"(\d+(?:\.\d+)?)"
                r"\s*(?:mbps|mb/s)",
                value,
                re.IGNORECASE
            )

            if match:

                return float(
                    match.group(1)
                )

        return None


    def search(
        obj,
        path=""
    ):

        if isinstance(
            obj,
            dict
        ):

            for key, value in obj.items():

                key_text = (
                    str(
                        key
                    )
                    .lower()
                )

                new_path = (
                    f"{path}.{key_text}"
                    if path
                    else key_text
                )


                relevant = any(
                    word
                    in new_path

                    for word in [
                        "speed",
                        "download",
                        "mbps",
                        "broadband"
                    ]
                )


                percentage = any(
                    word
                    in new_path

                    for word in [
                        "percent",
                        "percentage",
                        "coverage",
                        "_pct"
                    ]
                )


                number = extract_number(
                    value
                )


                if (
                    relevant
                    and not percentage
                    and number is not None
                    and number > 0
                ):

                    candidates.append({
                        "path":
                            new_path,

                        "value":
                            number
                    })


                if isinstance(
                    value,
                    str
                ):

                    mbps_number = (
                        extract_number(
                            value
                        )
                    )

                    if (
                        mbps_number is not None
                        and mbps_number > 0
                    ):

                        candidates.append({
                            "path":
                                new_path,

                            "value":
                                mbps_number
                        })


                search(
                    value,
                    new_path
                )


        elif isinstance(
            obj,
            list
        ):

            for index, item in enumerate(
                obj
            ):

                search(
                    item,
                    f"{path}[{index}]"
                )


    search(
        result
    )


    if not candidates:

        raise RuntimeError(
            f"No Mbps value found "
            f"for {postcode}"
        )


    download_candidates = [

        item

        for item
        in candidates

        if "download"
        in item["path"]
    ]


    if download_candidates:

        speed = max(
            item[
                "value"
            ]

            for item
            in download_candidates
        )

    else:

        speed = max(
            item[
                "value"
            ]

            for item
            in candidates
        )


    return {
        "internet_download_mbps":
            float(
                speed
            ),

        "good_internet":
            float(
                speed
            )
            >= GOOD_INTERNET_MBPS
    }


# =========================================================
# MODEL
# =========================================================

def predict_growth_from_row(
    row
):

    input_data = {

        feature:
            row[
                feature
            ]

        for feature
        in MODEL_FEATURES
    }


    X_new = pd.DataFrame(
        [
            input_data
        ],
        columns=MODEL_FEATURES
    )


    prediction = (
        growth_model
        .predict(
            X_new
        )[0]
    )


    return float(
        prediction
    )


def add_model_predictions(
    candidate_df
):

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
        candidate_df[
            "price_data_7"
        ]
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
    mode,
    max_candidates=12
):

    candidates = df.copy()


    candidates[
        "postcode"
    ] = (
        candidates[
            "postcode"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )


    candidates = candidates[
        candidates[
            "price_data_7"
        ] > 0
    ].copy()


    candidates = (
        add_model_predictions(
            candidates
        )
    )


    candidates = (
        candidates
        .drop_duplicates(
            subset="postcode",
            keep="first"
        )
    )


    # =====================================================
    # BUY / INVEST
    # =====================================================

    if mode in [
        "buy",
        "invest"
    ]:

        candidates = candidates[
            candidates[
                "price_data_7"
            ]
            <= budget_value * 1.20
        ]


        candidates = (
            candidates
            .sort_values(
                "predicted_growth",
                ascending=False
            )
            .head(
                max_candidates
            )
        )


        return candidates


    # =====================================================
    # RENT
    # =====================================================

    if mode == "rent":

        # Do NOT compare £400/week against
        # property purchase prices.

        # Use a reasonably broad candidate
        # pool and check actual rent later.
        candidates = (
            candidates
            .sort_values(
                [
                    "price_data_7",
                    "predicted_growth"
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .head(
                max_candidates
            )
        )


        return candidates


    return candidates.head(
        max_candidates
    )


# =========================================================
# ENRICH CANDIDATE
# =========================================================

def enrich_candidate(
    row,
    bedrooms_value,
    mode
):

    area_postcode = clean_postcode(
        row[
            "postcode"
        ]
    )


    output = (
        row.to_dict()
    )


    output[
        "postcode"
    ] = area_postcode


    output[
        "mode"
    ] = mode


    detail_postcode = (
        resolve_full_postcode(
            area_postcode
        )
    )


    output[
        "detail_postcode"
    ] = detail_postcode


    # =====================================================
    # PURCHASE PRICE
    # =====================================================

    if mode in [
        "buy",
        "invest"
    ]:

        try:

            output[
                "live_price"
            ] = fetch_live_price(
                area_postcode,
                bedrooms_value
            )

        except Exception as e:

            print(
                f"[PRICE ERROR] "
                f"{area_postcode}: "
                f"{e}"
            )

            output[
                "live_price"
            ] = float(
                row[
                    "price_data_7"
                ]
            )

    else:

        output[
            "live_price"
        ] = float(
            row[
                "price_data_7"
            ]
        )


    # =====================================================
    # RENT
    # =====================================================

    output[
        "live_rent"
    ] = None


    if mode in [
        "rent",
        "invest"
    ]:

        try:

            output[
                "live_rent"
            ] = fetch_live_rent(
                area_postcode,
                bedrooms_value
            )

        except Exception as first_error:

            try:

                if detail_postcode:

                    output[
                        "live_rent"
                    ] = fetch_live_rent(
                        detail_postcode,
                        bedrooms_value
                    )

                else:

                    raise first_error

            except Exception as e:

                print(
                    f"[RENT ERROR] "
                    f"{area_postcode}: "
                    f"{e}"
                )

                output[
                    "live_rent"
                ] = None


    # =====================================================
    # GROSS YIELD
    # =====================================================

    output[
        "gross_yield"
    ] = None


    if (
        mode == "invest"
        and output.get(
            "live_rent"
        ) is not None
        and output.get(
            "live_price"
        )
    ):

        output[
            "gross_yield"
        ] = (
            float(
                output[
                    "live_rent"
                ]
            )
            * 52
            /
            float(
                output[
                    "live_price"
                ]
            )
            * 100
        )


    # =====================================================
    # AVERAGE BEDROOMS
    # =====================================================

    try:

        avg_bedrooms = (
            fetch_average_bedrooms(
                area_postcode,
                mode
            )
        )


        if (
            avg_bedrooms is None
            and detail_postcode
        ):

            avg_bedrooms = (
                fetch_average_bedrooms(
                    detail_postcode,
                    mode
                )
            )


        output[
            "average_bedrooms"
        ] = avg_bedrooms


    except Exception as e:

        print(
            f"[BEDROOM ERROR] "
            f"{area_postcode}: "
            f"{e}"
        )

        output[
            "average_bedrooms"
        ] = None


    # =====================================================
    # CRIME
    # =====================================================

    try:

        output.update(
            fetch_crime(
                area_postcode
            )
        )

    except Exception as first_error:

        try:

            if detail_postcode:

                output.update(
                    fetch_crime(
                        detail_postcode
                    )
                )

            else:

                raise first_error

        except Exception as e:

            print(
                f"[CRIME ERROR] "
                f"{area_postcode}: "
                f"{e}"
            )

            output[
                "crime_rate"
            ] = np.nan

            output[
                "crime_rating"
            ] = "Unavailable"


    # =====================================================
    # SCHOOL
    # =====================================================

    school_postcode = (
        detail_postcode
        or area_postcode
    )


    try:

        output.update(
            fetch_school_data(
                school_postcode
            )
        )

    except Exception as e:

        print(
            f"[SCHOOL ERROR] "
            f"{area_postcode}: "
            f"{e}"
        )

        output.update({
            "school_nearby":
                None,

            "school_distance":
                None,

            "nearest_school":
                None
        })


    # =====================================================
    # RESTAURANTS
    # =====================================================

    if detail_postcode:

        try:

            output.update(
                fetch_restaurant_data(
                    detail_postcode
                )
            )

        except Exception as e:

            print(
                f"[RESTAURANT ERROR] "
                f"{area_postcode}: "
                f"{e}"
            )

            output.update({
                "restaurants_nearby":
                    None,

                "restaurant_count":
                    None,

                "nearest_restaurant_distance":
                    None
            })

    else:

        output.update({
            "restaurants_nearby":
                None,

            "restaurant_count":
                None,

            "nearest_restaurant_distance":
                None
        })


    # =====================================================
    # FLOOD
    # =====================================================

    if detail_postcode:

        try:

            output.update(
                fetch_flood_data(
                    detail_postcode
                )
            )

        except Exception as e:

            print(
                f"[FLOOD ERROR] "
                f"{area_postcode}: "
                f"{e}"
            )

            output.update({
                "flood_risk":
                    None,

                "low_flood_risk":
                    None
            })

    else:

        output.update({
            "flood_risk":
                None,

            "low_flood_risk":
                None
        })


    # =====================================================
    # INTERNET
    # =====================================================

    if detail_postcode:

        try:

            internet_data = (
                fetch_internet_data(
                    detail_postcode
                )
            )

            output.update(
                internet_data
            )

        except Exception as e:

            print(
                f"[INTERNET ERROR] "
                f"{area_postcode} "
                f"({detail_postcode}): "
                f"{e}"
            )

            output.update({
                "internet_download_mbps":
                    None,

                "good_internet":
                    None
            })

    else:

        output.update({
            "internet_download_mbps":
                None,

            "good_internet":
                None
        })


    # =====================================================
    # FUTURE PRICE
    # =====================================================

    output[
        "predicted_future_price"
    ] = (
        float(
            output[
                "live_price"
            ]
        )
        *
        (
            1
            +
            float(
                output[
                    "predicted_growth"
                ]
            )
            / 100
        )
    )


    return output


# =========================================================
# PARALLEL ENRICHMENT
# =========================================================

def enrich_shortlist_parallel(
    shortlist,
    bedrooms_value,
    mode,
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
                bedrooms_value,
                mode
            ):
            row[
                "postcode"
            ]

            for row
            in rows
        }


        for future in as_completed(
            futures
        ):

            postcode = futures[
                future
            ]

            try:

                enriched.append(
                    future.result()
                )

            except Exception as e:

                print(
                    f"Could not load "
                    f"{postcode}: "
                    f"{e}"
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
        ~np.isnan(
            values
        )
    ]


    if len(
        valid
    ) == 0:

        return np.ones(
            len(
                values
            )
        )


    low = np.nanmin(
        values
    )

    high = np.nanmax(
        values
    )


    if high == low:

        return np.ones(
            len(
                values
            )
        )


    if higher_is_better:

        return (
            values
            - low
        ) / (
            high
            - low
        )


    return (
        high
        - values
    ) / (
        high
        - low
    )


# =========================================================
# AMENITY ICON
# =========================================================

def amenity_icon(
    value
):

    if value is True:

        return "✅"

    if value is False:

        return "❌"

    return "—"


# =========================================================
# MODE
# =========================================================

property_mode = widgets.Dropdown(

    options=[
        (
            "Buy",
            "buy"
        ),
        (
            "Rent",
            "rent"
        ),
        (
            "Invest",
            "invest"
        )
    ],

    value="buy",

    description="",

    layout=widgets.Layout(
        width="140px"
    )
)


# =========================================================
# BUDGET
# =========================================================

budget = widgets.Text(

    value="",

    placeholder=(
        "Enter maximum property value (£)"
    ),

    description="",

    layout=widgets.Layout(
        width="320px"
    )
)


def get_budget_value():

    try:

        cleaned = (
            budget.value
            .replace(
                "£",
                ""
            )
            .replace(
                ",",
                ""
            )
            .strip()
        )


        value = float(
            cleaned
        )


        if value <= 0:

            raise ValueError


        return value


    except Exception:

        if (
            property_mode.value
            == "rent"
        ):

            raise ValueError(
                "Enter a valid weekly rent, "
                "for example 400."
            )

        raise ValueError(
            "Enter a valid property budget."
        )


# =========================================================
# BEDROOM FILTER
# =========================================================

bedrooms = widgets.Dropdown(

    options=[
        (
            "Any",
            0
        ),
        (
            "1 bedroom",
            1
        ),
        (
            "2 bedrooms",
            2
        ),
        (
            "3 bedrooms",
            3
        ),
        (
            "4 bedrooms",
            4
        ),
        (
            "5+ bedrooms",
            5
        )
    ],

    value=0,

    description="Bedrooms:",

    style={
        "description_width":
            "90px"
    },

    layout=widgets.Layout(
        width="270px"
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


restaurants_checkbox = widgets.Checkbox(
    value=False,
    description="Restaurants nearby",
    indent=False
)


flood_checkbox = widgets.Checkbox(
    value=False,
    description="Low flood risk",
    indent=False
)


internet_checkbox = widgets.Checkbox(
    value=False,
    description="Good internet speed",
    indent=False
)


amenity_box = widgets.HBox(

    [
        school_checkbox,
        restaurants_checkbox,
        flood_checkbox,
        internet_checkbox
    ],

    layout=widgets.Layout(
        gap="25px",
        flex_flow="row wrap"
    )
)


# =========================================================
# WEIGHTS
# =========================================================

names = {

    "price":
        "Purchase Price",

    "growth":
        "Predicted Growth",

    "crime":
        "Crime & Safety"
}


defaults = {

    "price":
        40,

    "growth":
        40,

    "crime":
        20
}


keys = list(
    names.keys()
)


sliders = {}
numbers = {}
lock_buttons = {}
weight_labels = {}

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
# BUILD WEIGHT CONTROLS
# =========================================================

for key in keys:

    label = widgets.HTML(

        value=(
            f"<b>{names[key]}</b>"
        ),

        layout=widgets.Layout(
            width="150px",
            min_width="150px"
        )
    )


    weight_labels[
        key
    ] = label


    slider = widgets.IntSlider(

        value=defaults[
            key
        ],

        min=0,
        max=100,
        step=1,

        readout=False,

        continuous_update=True,

        layout=widgets.Layout(
            width="100%",
            min_width="150px",
            flex="1 1 auto"
        )
    )


    number = widgets.BoundedIntText(

        value=defaults[
            key
        ],

        min=0,
        max=100,
        step=1,

        layout=widgets.Layout(
            width="70px"
        )
    )


    percent = widgets.HTML(
        "%"
    )


    lock_button = widgets.ToggleButton(

        value=False,

        description="🔓",

        tooltip="Lock this weight",

        layout=widgets.Layout(
            width="50px"
        )
    )


    sliders[
        key
    ] = slider


    numbers[
        key
    ] = number


    lock_buttons[
        key
    ] = lock_button


    weight_rows.append(

        widgets.HBox(

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
                gap="7px"
            )
        )
    )


preferences_box = widgets.VBox(

    weight_rows,

    layout=widgets.Layout(
        width="100%",
        gap="7px"
    )
)


# =========================================================
# LOCKS
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
# TOTAL
# =========================================================

def refresh_total():

    total = sum(
        numbers[
            k
        ].value

        for k
        in keys
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


# =========================================================
# DONUT CHART
# =========================================================

def refresh_chart():

    values = [

        numbers[
            k
        ].value

        for k
        in keys
    ]


    fig = go.Figure(

        data=[

            go.Pie(

                labels=[
                    names[
                        k
                    ]

                    for k
                    in keys
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

        height=320,

        margin=dict(
            l=10,
            r=10,
            t=45,
            b=30
        ),

        legend=dict(
            orientation="h",
            y=-0.05,
            x=0.5,
            xanchor="center"
        )
    )


    with chart_output:

        clear_output(
            wait=True
        )

        display(
            fig
        )


# =========================================================
# MODE UI
# =========================================================

def update_mode_ui(
    change=None
):

    mode = (
        property_mode.value
    )


    if mode == "rent":

        budget.placeholder = (
            "Enter maximum rent per week (£)"
        )

        names[
            "price"
        ] = "Weekly Rent"


    elif mode == "invest":

        budget.placeholder = (
            "Enter maximum investment value (£)"
        )

        names[
            "price"
        ] = "Investment Price"


    else:

        budget.placeholder = (
            "Enter maximum property value (£)"
        )

        names[
            "price"
        ] = "Purchase Price"


    weight_labels[
        "price"
    ].value = (
        f"<b>{names['price']}</b>"
    )


    refresh_chart()


property_mode.observe(
    update_mode_ui,
    names="value"
)


# =========================================================
# WEIGHT REDISTRIBUTION
# =========================================================

def change_weight(
    changed_key,
    new_value
):

    global busy


    if busy:

        return


    if lock_buttons[
        changed_key
    ].value:

        busy = True

        try:

            sliders[
                changed_key
            ].value = (
                numbers[
                    changed_key
                ].value
            )

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

        for k
        in keys

        if lock_buttons[
            k
        ].value
    ]


    unlocked_keys = [

        k

        for k
        in keys

        if not lock_buttons[
            k
        ].value
    ]


    locked_total = sum(

        numbers[
            k
        ].value

        for k
        in locked_keys
    )


    available_total = (
        100
        - locked_total
    )


    if len(
        unlocked_keys
    ) == 1:

        only_key = (
            unlocked_keys[
                0
            ]
        )


        busy = True


        try:

            numbers[
                only_key
            ].value = (
                available_total
            )


            sliders[
                only_key
            ].value = (
                available_total
            )


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

        for k
        in unlocked_keys

        if k != changed_key
    ]


    remaining = (
        available_total
        - new_value
    )


    old_values = {

        k:
            numbers[
                k
            ].value

        for k
        in other_unlocked
    }


    old_total = sum(
        old_values.values()
    )


    if old_total == 0:

        exact = {

            k:
                remaining
                / len(
                    other_unlocked
                )

            for k
            in other_unlocked
        }

    else:

        exact = {

            k:
                remaining
                * old_values[
                    k
                ]
                / old_total

            for k
            in other_unlocked
        }


    allocated = {

        k:
            int(
                exact[
                    k
                ]
            )

        for k
        in other_unlocked
    }


    leftover = (
        remaining
        - sum(
            allocated.values()
        )
    )


    order = sorted(

        other_unlocked,

        key=lambda k:
            exact[
                k
            ]
            - allocated[
                k
            ],

        reverse=True
    )


    for k in order[
        :leftover
    ]:

        allocated[
            k
        ] += 1


    busy = True


    try:

        numbers[
            changed_key
        ].value = (
            new_value
        )


        sliders[
            changed_key
        ].value = (
            new_value
        )


        for k in other_unlocked:

            numbers[
                k
            ].value = (
                allocated[
                    k
                ]
            )


            sliders[
                k
            ].value = (
                allocated[
                    k
                ]
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
        change[
            "name"
        ] == "value"
        and not busy
    ):

        change_weight(
            key,
            change[
                "new"
            ]
        )


def on_number(
    change,
    key
):

    if (
        change[
            "name"
        ] == "value"
        and not busy
    ):

        change_weight(
            key,
            change[
                "new"
            ]
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
# SCORE
# =========================================================

def score_candidates(
    candidates,
    budget_value,
    mode
):

    if not candidates:

        return []


    # =====================================================
    # RENT
    # =====================================================

    if mode == "rent":

        candidates = [

            c

            for c
            in candidates

            if c.get(
                "live_rent"
            ) is not None
        ]


        candidates = [

            c

            for c
            in candidates

            if float(
                c[
                    "live_rent"
                ]
            )
            <= budget_value
        ]


        costs = [

            float(
                c[
                    "live_rent"
                ]
            )

            for c
            in candidates
        ]


    # =====================================================
    # BUY / INVEST
    # =====================================================

    else:

        candidates = [

            c

            for c
            in candidates

            if c.get(
                "live_price"
            ) is not None
        ]


        candidates = [

            c

            for c
            in candidates

            if float(
                c[
                    "live_price"
                ]
            )
            <= budget_value
        ]


        costs = [

            float(
                c[
                    "live_price"
                ]
            )

            for c
            in candidates
        ]


    if not candidates:

        return []


    growth_values = [

        float(
            c[
                "predicted_growth"
            ]
        )

        for c
        in candidates
    ]


    valid_crimes = [

        float(
            c[
                "crime_rate"
            ]
        )

        for c
        in candidates

        if not pd.isna(
            c.get(
                "crime_rate",
                np.nan
            )
        )
    ]


    crime_fallback = (

        np.median(
            valid_crimes
        )

        if valid_crimes

        else 0
    )


    crime_values = [

        float(
            c[
                "crime_rate"
            ]
        )

        if not pd.isna(
            c.get(
                "crime_rate",
                np.nan
            )
        )

        else crime_fallback

        for c
        in candidates
    ]


    cost_score = normalise(
        costs,
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
            numbers[
                key
            ].value
            / 100

        for key
        in keys
    }


    scored = []


    for i, candidate in enumerate(
        candidates
    ):


        if (
            school_checkbox.value
            and candidate.get(
                "school_nearby"
            ) is not True
        ):

            continue


        if (
            restaurants_checkbox.value
            and candidate.get(
                "restaurants_nearby"
            ) is not True
        ):

            continue


        if (
            flood_checkbox.value
            and candidate.get(
                "low_flood_risk"
            ) is not True
        ):

            continue


        if (
            internet_checkbox.value
            and candidate.get(
                "good_internet"
            ) is not True
        ):

            continue


        score = 100 * (

            weights[
                "price"
            ]
            * cost_score[
                i
            ]

            +

            weights[
                "growth"
            ]
            * growth_score[
                i
            ]

            +

            weights[
                "crime"
            ]
            * crime_score[
                i
            ]
        )


        result = (
            candidate.copy()
        )


        result[
            "score"
        ] = float(
            score
        )


        scored.append(
            result
        )


    return sorted(

        scored,

        key=lambda x:
            x[
                "score"
            ],

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

            description=p[
                "postcode"
            ],

            indent=False
        )


        comparison_checkboxes.append(
            (
                checkbox,
                p[
                    "postcode"
                ]
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

def recommend(_=None):

    global current_recommendations


    # =====================================================
    # READ MODE / BUDGET
    # =====================================================

    mode = (
        property_mode.value
    )


    try:

        budget_value = (
            get_budget_value()
        )

    except ValueError as e:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    f"""
                    <div style="
                        color:#b42318;
                        font-weight:bold;
                        padding:10px;
                    ">
                        {e}
                    </div>
                    """
                )
            )

        return


    # =====================================================
    # LOADING MESSAGE
    # =====================================================

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
                        Finding matching areas...
                    </b>
                </div>
                """
            )
        )


    # Rent needs more candidates because
    # the weekly rent isn't stored in CSV.
    max_candidates = (
        16
        if mode == "rent"
        else 10
    )


    shortlist = (
        create_initial_shortlist(
            budget_value,
            mode,
            max_candidates=max_candidates
        )
    )


    if shortlist.empty:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    """
                    <b style="color:#b42318">
                        No areas found for
                        this budget.
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

        # =====================================================
    # FETCH LIVE API DATA
    # =====================================================

    enriched = enrich_shortlist_parallel(
        shortlist,
        bedrooms.value,
        mode,
        max_workers=4
    )


    if not enriched:

        with results_output:

            clear_output()

            display(
                widgets.HTML(
                    """
                    <b style="
                        color:#b42318;
                    ">
                        No live API data could be loaded.
                        Check the notebook output for
                        API error messages.
                    </b>
                    """
                )
            )

        return


    # =====================================================
    # SCORE RESULTS
    # =====================================================

    ranked = score_candidates(
        enriched,
        budget_value,
        mode
    )


    current_recommendations = (
        ranked[:5]
    )


    update_comparison_options()


    # =====================================================
    # NO MATCHES
    # =====================================================

    if not current_recommendations:

        with results_output:

            clear_output()


            if mode == "rent":

                message = (
                    f"No rental areas matched "
                    f"within £{budget_value:,.0f}/week."
                )

            elif mode == "invest":

                message = (
                    f"No investment areas matched "
                    f"within £{budget_value:,.0f}."
                )

            else:

                message = (
                    f"No buying areas matched "
                    f"within £{budget_value:,.0f}."
                )


            display(
                widgets.HTML(
                    f"""
                    <b style="
                        color:#b42318;
                    ">
                        {message}
                    </b>
                    """
                )
            )

        return

    # =====================================================
    # RESULT CARDS
    # =====================================================

    cards = []


    for rank, p in enumerate(
        current_recommendations,
        start=1
    ):


        price = float(
            p.get(
                "live_price",
                0
            )
        )


        growth = float(
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


        # =================================================
        # BEDROOM DISPLAY
        # =================================================

        bedrooms_display = (
            display_bedrooms(
                p.get(
                    "average_bedrooms"
                )
            )
        )


        # =================================================
        # MODE VALUES
        # =================================================

        if mode == "rent":

            rent = p.get(
                "live_rent"
            )


            if rent is not None:

                rent = float(
                    rent
                )


                cost_label = (
                    "Weekly rent"
                )


                cost_display = (
                    f"£{rent:,.0f}/week"
                )


                second_label = (
                    "Approx. monthly rent"
                )


                second_display = (
                    f"£{rent * 52 / 12:,.0f}/month"
                )

            else:

                cost_label = (
                    "Weekly rent"
                )


                cost_display = "—"


                second_label = (
                    "Approx. monthly rent"
                )


                second_display = "—"


            third_label = (
                "Predicted area growth"
            )


            third_display = (
                f"{growth:.2f}%"
            )


            mode_name = "RENT"


        elif mode == "invest":

            cost_label = (
                "Investment price"
            )


            cost_display = (
                f"£{price:,.0f}"
            )


            gross_yield = (
                p.get(
                    "gross_yield"
                )
            )


            second_label = (
                "Gross rental yield"
            )


            second_display = (

                "—"

                if gross_yield is None

                else
                f"{float(gross_yield):.2f}%"
            )


            third_label = (
                "Predicted growth"
            )


            third_display = (
                f"{growth:.2f}%"
            )


            mode_name = "INVEST"


        else:

            cost_label = (
                "Current asking price"
            )


            cost_display = (
                f"£{price:,.0f}"
            )


            second_label = (
                "Predicted growth"
            )


            second_display = (
                f"{growth:.2f}%"
            )


            third_label = (
                "Predicted future price"
            )


            third_display = (
                f"£{future_price:,.0f}"
            )


            mode_name = "BUY"


        # =================================================
        # AMENITIES
        # =================================================

        school_icon = amenity_icon(
            p.get(
                "school_nearby"
            )
        )


        restaurant_icon = amenity_icon(
            p.get(
                "restaurants_nearby"
            )
        )


        flood_icon = amenity_icon(
            p.get(
                "low_flood_risk"
            )
        )


        internet_speed = (
            p.get(
                "internet_download_mbps"
            )
        )


        internet_display = (

            "—"

            if internet_speed is None

            else
            f"{float(internet_speed):.0f} Mbps"
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


        if school_distance is not None:

            school_text = (
                f"{nearest_school} "
                f"({float(school_distance):.1f} mi)"
            )

        else:

            school_text = (
                nearest_school
            )


        restaurant_count = (
            p.get(
                "restaurant_count"
            )
        )


        restaurant_text = (

            "Unavailable"

            if restaurant_count is None

            else
            f"{restaurant_count} found"
        )


        flood_text = (

            p.get(
                "flood_risk"
            )

            or "Unavailable"
        )


        cards.append(
            f"""
            <div style="
                border:1px solid #d9d9d9;
                border-radius:12px;
                padding:18px;
                margin:0 0 14px 0;
                background:#ffffff;
                color:#222222;
            ">


                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    gap:15px;
                ">

                    <div>

                        <div style="
                            color:#666;
                            font-size:12px;
                            font-weight:bold;
                        ">
                            RANK #{rank}
                            &nbsp; · &nbsp;
                            {mode_name}
                        </div>

                        <div style="
                            color:#111;
                            font-size:23px;
                            font-weight:bold;
                        ">
                            {p['postcode']}
                        </div>

                    </div>


                    <div style="
                        color:#111;
                        font-size:25px;
                        font-weight:bold;
                    ">
                        {score:.0f}/100
                    </div>

                </div>


                <hr style="
                    border:none;
                    border-top:1px solid #ddd;
                ">


                <div style="
                    display:grid;
                    grid-template-columns:
                    repeat(
                        auto-fit,
                        minmax(150px,1fr)
                    );
                    gap:18px;
                ">


                    <div>

                        <span style="color:#666">
                            {cost_label}
                        </span>

                        <br>

                        <b style="color:#111">
                            {cost_display}
                        </b>

                    </div>


                    <div>

                        <span style="color:#666">
                            {second_label}
                        </span>

                        <br>

                        <b style="color:#111">
                            {second_display}
                        </b>

                    </div>


                    <div>

                        <span style="color:#666">
                            {third_label}
                        </span>

                        <br>

                        <b style="color:#111">
                            {third_display}
                        </b>

                    </div>


                    <div>

                        <span style="color:#666">
                            Crime
                        </span>

                        <br>

                        <b style="color:#111">
                            {
                                p.get(
                                    "crime_rating",
                                    "Unavailable"
                                )
                            }
                        </b>

                    </div>


                    <div>

                        <span style="color:#666">
                            Average bedrooms
                        </span>

                        <br>

                        <b style="color:#111">
                            🛏️ {bedrooms_display}
                        </b>

                    </div>

                </div>


                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:25px;
                    margin-top:18px;
                    padding-top:14px;
                    border-top:1px solid #eee;
                    color:#222;
                ">

                    <div>

                        {school_icon}

                        <b style="color:#111">
                            School
                        </b>

                    </div>


                    <div>

                        {restaurant_icon}

                        <b style="color:#111">
                            Restaurants
                        </b>

                    </div>


                    <div>

                        {flood_icon}

                        <b style="color:#111">
                            Low flood risk
                        </b>

                    </div>


                    <div>

                        🌐

                        <b style="color:#111">
                            Internet:
                        </b>

                        <span style="
                            color:#111;
                            font-weight:600;
                        ">
                            {internet_display}
                        </span>

                    </div>

                </div>


                <div style="
                    color:#666;
                    margin-top:12px;
                    font-size:12px;
                    line-height:1.8;
                ">

                    Nearest school:
                    {school_text}

                    <br>

                    Restaurants:
                    {restaurant_text}

                    <br>

                    Flood risk:
                    {flood_text}

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
                "".join(
                    cards
                )
            )
        )


# =========================================================
# COMPARE
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
                    <b style="
                        color:#b42318;
                    ">
                        Please select at least
                        two areas.
                    </b>
                    """
                )
            )

            return


        selected = [

            p

            for p
            in current_recommendations

            if p[
                "postcode"
            ]
            in selected_codes
        ]


        mode = (
            property_mode.value
        )


        if mode == "rent":

            cost_header = (
                "Weekly Rent"
            )

            extra_header = (
                "Monthly Rent"
            )


        elif mode == "invest":

            cost_header = (
                "Investment Price"
            )

            extra_header = (
                "Gross Yield"
            )


        else:

            cost_header = (
                "Price"
            )

            extra_header = (
                "Future Price"
            )


        rows = ""


        for p in selected:

            price = float(
                p.get(
                    "live_price",
                    0
                )
            )


            growth = float(
                p.get(
                    "predicted_growth",
                    0
                )
            )


            score = float(
                p.get(
                    "score",
                    0
                )
            )


            bedrooms_display = (
                display_bedrooms(
                    p.get(
                        "average_bedrooms"
                    )
                )
            )


            # =============================================
            # MODE VALUES
            # =============================================

            if mode == "rent":

                rent = (
                    p.get(
                        "live_rent"
                    )
                )


                if rent is None:

                    cost_display = "—"

                    extra_display = "—"

                else:

                    rent = float(
                        rent
                    )


                    cost_display = (
                        f"£{rent:,.0f}/wk"
                    )


                    extra_display = (
                        f"£{rent * 52 / 12:,.0f}/mo"
                    )


            elif mode == "invest":

                cost_display = (
                    f"£{price:,.0f}"
                )


                gross_yield = (
                    p.get(
                        "gross_yield"
                    )
                )


                extra_display = (

                    "—"

                    if gross_yield is None

                    else
                    f"{float(gross_yield):.2f}%"
                )


            else:

                future_price = float(
                    p.get(
                        "predicted_future_price",
                        0
                    )
                )


                cost_display = (
                    f"£{price:,.0f}"
                )


                extra_display = (
                    f"£{future_price:,.0f}"
                )


            school_icon = (
                amenity_icon(
                    p.get(
                        "school_nearby"
                    )
                )
            )


            restaurant_icon = (
                amenity_icon(
                    p.get(
                        "restaurants_nearby"
                    )
                )
            )


            flood_icon = (
                amenity_icon(
                    p.get(
                        "low_flood_risk"
                    )
                )
            )


            internet_speed = (
                p.get(
                    "internet_download_mbps"
                )
            )


            internet_display = (

                "—"

                if internet_speed is None

                else
                f"{float(internet_speed):.0f} Mbps"
            )


            rows += f"""
            <tr style="
                border-bottom:1px solid #ddd;
                color:#222;
            ">

                <td style="padding:10px">

                    <b style="color:#111">
                        {p['postcode']}
                    </b>

                </td>


                <td style="padding:10px">
                    {cost_display}
                </td>


                <td style="padding:10px">
                    {growth:.2f}%
                </td>


                <td style="padding:10px">
                    {extra_display}
                </td>


                <td style="padding:10px">

                    {
                        p.get(
                            "crime_rating",
                            "Unavailable"
                        )
                    }

                </td>


                <td style="
                    padding:10px;
                    text-align:center;
                    color:#111;
                    font-weight:600;
                ">
                    🛏️ {bedrooms_display}
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
                    {restaurant_icon}
                </td>


                <td style="
                    padding:10px;
                    text-align:center;
                    font-size:20px;
                ">
                    {flood_icon}
                </td>


                <td style="
                    padding:10px;
                    text-align:center;
                    color:#111;
                    font-weight:600;
                ">
                    {internet_display}
                </td>


                <td style="padding:10px">

                    <b style="color:#111">
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
                        min-width:1200px;
                        border-collapse:collapse;
                        font-size:13px;
                        background:#ffffff;
                        color:#222222;
                    ">


                        <thead>

                            <tr style="
                                background:#eeeeee;
                                color:#111111;
                                border-bottom:
                                2px solid #bbb;
                            ">

                                <th style="padding:10px">
                                    Area
                                </th>

                                <th style="padding:10px">
                                    {cost_header}
                                </th>

                                <th style="padding:10px">
                                    Growth
                                </th>

                                <th style="padding:10px">
                                    {extra_header}
                                </th>

                                <th style="padding:10px">
                                    Crime
                                </th>

                                <th style="padding:10px">
                                    Avg. Bedrooms
                                </th>

                                <th style="padding:10px">
                                    School
                                </th>

                                <th style="padding:10px">
                                    Restaurants
                                </th>

                                <th style="padding:10px">
                                    Low Flood Risk
                                </th>

                                <th style="padding:10px">
                                    Internet Speed
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

                    🛏️ Average bedrooms rounded
                    to the nearest whole bedroom

                    &nbsp;&nbsp;

                    ✅ meets requirement

                    &nbsp;&nbsp;

                    ❌ does not meet requirement

                    &nbsp;&nbsp;

                    — data unavailable

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
        width="220px",
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
# APP LAYOUT
# =========================================================

app = widgets.VBox(

    [

        widgets.HTML(
            """
            <div>

                <h2 style="
                    margin:0 0 3px 0;
                ">
                    PropertyFinder
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
                property_mode,
                bedrooms
            ],

            layout=widgets.Layout(
                gap="12px",
                flex_flow="row wrap",
                align_items="center"
            )
        ),


        widgets.HTML(
            """
            <h3>
                Nearby / area requirements
            </h3>
            """
        ),


        amenity_box,


        widgets.HTML(
            """
            <small style="color:#777">

                Select any requirements
                you want the recommended
                area to satisfy.

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

                🔒 Lock a preference so
                changing another slider
                does not alter it.

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
                your selected property mode,
                budget and priorities.

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

update_mode_ui()

display(
    app
)