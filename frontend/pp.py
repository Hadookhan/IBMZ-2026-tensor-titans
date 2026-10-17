import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go

# =========================
# HomeMatch AI - Demo Data
# =========================


properties = [
    {
        "postcode": "M1 1AE",
        "price": 235000,
        "growth": 4.8,
        "crime": 42,
        "school": 0.6,
        "hospital": 1.1,
        "work": 1.4,
        "bedrooms": 2
    },
    {
        "postcode": "M20 2YY",
        "price": 385000,
        "growth": 5.6,
        "crime": 24,
        "school": 0.4,
        "hospital": 2.0,
        "work": 3.2,
        "bedrooms": 3
    },
    {
        "postcode": "M14 6PL",
        "price": 265000,
        "growth": 3.9,
        "crime": 35,
        "school": 0.8,
        "hospital": 0.9,
        "work": 1.8,
        "bedrooms": 2
    },
]


names = {
    "price": "Sale Price",
    "growth": "Growth Rate",
    "crime": "Crime & Safety",
    "school": "Distance to School",
    "hospital": "Distance to Hospital",
    "work": "Distance to Workplace",
}

defaults = {
    "price": 25, "growth": 20, "crime": 15,
    "school": 10, "hospital": 10, "work": 20
}
keys = list(names.keys())

# =========================
# Page heading and inputs
# =========================

title = widgets.HTML("""
<h2 style="margin-bottom:4px">HomeMatch AI</h2>
<p style="color:gray;margin-top:0">
Property preference comparison — illustrative demo data
</p>
""")

budget = widgets.IntText(
    value=400000,
    description="Budget (£):",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="250px")
)

postcodes = widgets.Text(
    value="M1 1AE, M20 2YY, M14 6PL",
    description="Postcodes:",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="100%", max_width="650px")
)

school_name = widgets.Text(
    description="School:",
    placeholder="Enter school name",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="100%", max_width="450px")
)

hospital_name = widgets.Text(
    description="Hospital:",
    placeholder="Enter hospital name",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="100%", max_width="450px")
)

workplace_name = widgets.Text(
    description="Workplace:",
    placeholder="Enter workplace or postcode",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="100%", max_width="450px")
)


# Bedroom preference
bedrooms = widgets.Dropdown(
    options=[
        ("Any", 0),
        ("1 bedroom", 1),
        ("2 bedrooms", 2),
        ("3 bedrooms", 3),
        ("4 bedrooms", 4),
        ("5+ bedrooms", 5)
    ],
    value=2,
    description="Bedrooms:",
    style={"description_width": "100px"},
    layout=widgets.Layout(width="250px")
)

# =========================
# Six preference controls
# =========================

sliders = {}
numbers = {}
preference_rows = []

for key in keys:
    slider = widgets.IntSlider(
        value=defaults[key],
        min=0,
        max=100,
        step=1,
        readout=False,
        description="",
        continuous_update=True,
        layout=widgets.Layout(
            width="100%",
            min_width="80px",
            flex="1 1 auto"
        )
    )

    number = widgets.BoundedIntText(
        value=defaults[key],
        min=0,
        max=100,
        step=1,
        description="%",
        style={"description_width": "18px"},
        layout=widgets.Layout(
            width="100px",
            min_width="100px",
            flex="0 0 100px"
        )
    )

    label = widgets.HTML(
        value=f"<div style='width:175px'>{names[key]}</div>",
        layout=widgets.Layout(
            width="175px",
            min_width="175px",
            flex="0 0 175px"
        )
    )

    row = widgets.HBox(
        [label, slider, number],
        layout=widgets.Layout(
            width="100%",
            min_width="0px",
            align_items="center",
            gap="10px",
            margin="3px 0"
        )
    )

    sliders[key] = slider
    numbers[key] = number
    preference_rows.append(row)

preferences_box = widgets.VBox(
    preference_rows,
    layout=widgets.Layout(
        width="100%",
        min_width="0px",
        max_width="1000px",
        gap="5px"
    )
)

total_label = widgets.HTML()
chart_output = widgets.Output()
results_output = widgets.Output()

# =========================
# Preference synchronisation
# =========================

busy = False

def refresh_total():
    total = sum(numbers[k].value for k in keys)
    colour = "#16803c" if total == 100 else "#b42318"
    total_label.value = (
        f"<b style='color:{colour}'>"
        f"Total preference: {total}% / 100%</b>"
    )

def refresh_chart():
    values = [numbers[k].value for k in keys]

    fig = go.Figure(data=[go.Pie(
        labels=[names[k] for k in keys],
        values=values,
        hole=0.58,
        sort=False,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}%<extra></extra>"
    )])

    fig.update_layout(
        title="Your Preference Weights",
        height=430,
        autosize=True,
        margin=dict(l=15, r=15, t=55, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    with chart_output:
        clear_output(wait=True)
        display(fig)

def change_weight(changed_key, new_value):
    global busy

    if busy:
        return

    new_value = max(0, min(100, int(new_value)))
    other_keys = [k for k in keys if k != changed_key]
    remaining = 100 - new_value

    old_values = {
        k: max(0, int(numbers[k].value))
        for k in other_keys
    }
    old_total = sum(old_values.values())

    if old_total == 0:
        exact = {
            k: remaining / len(other_keys)
            for k in other_keys
        }
    else:
        exact = {
            k: remaining * old_values[k] / old_total
            for k in other_keys
        }

    allocated = {k: int(exact[k]) for k in other_keys}
    leftover = remaining - sum(allocated.values())

    order = sorted(
        other_keys,
        key=lambda k: exact[k] - allocated[k],
        reverse=True
    )

    for k in order[:leftover]:
        allocated[k] += 1

    busy = True
    try:
        for k in keys:
            value = new_value if k == changed_key else allocated[k]
            numbers[k].value = value
            sliders[k].value = value
    finally:
        busy = False

    refresh_total()
    refresh_chart()

def slider_changed(change, key):
    if change["name"] == "value" and not busy:
        change_weight(key, change["new"])

def number_changed(change, key):
    if change["name"] == "value" and not busy:
        change_weight(key, change["new"])

for key in keys:
    sliders[key].observe(
        lambda change, k=key: slider_changed(change, k),
        names="value"
    )
    numbers[key].observe(
        lambda change, k=key: number_changed(change, k),
        names="value"
    )

# =========================
# Property comparison
# =========================

def normalise(values, higher_is_better=True):
    low, high = min(values), max(values)

    if low == high:
        return [1.0] * len(values)

    if higher_is_better:
        return [(v - low) / (high - low) for v in values]

    return [(high - v) / (high - low) for v in values]

def compare_properties(_=None):
    with results_output:
        clear_output(wait=True)

        requested = [
            p.strip().upper()
            for p in postcodes.value.split(",")
            if p.strip()
        ]

        property_lookup = {
            p["postcode"].upper(): p
            for p in properties
        }

        selected = [
            property_lookup[p]
            for p in requested
            if p in property_lookup
        ]

        if bedrooms.value != 0:
            if bedrooms.value == 5:
                selected = [
                    p for p in selected
                    if p["bedrooms"] >= 5
                ]
            else:
                selected = [
                    p for p in selected
                    if p["bedrooms"] == bedrooms.value
                ]


        missing = [p for p in requested if p not in property_lookup]

        if not selected:
            display(widgets.HTML(
                "<b style='color:#b42318'>"
                "No matching demo postcodes. Try M1 1AE, M20 2YY, M14 6PL."
                "</b>"
            ))
            return

        if missing:
            display(widgets.HTML(
                "<p>Not in demo dataset: "
                + ", ".join(missing)
                + "</p>"
            ))

        criteria = {
            "price": normalise(
                [p["price"] for p in selected], False
            ),
            "growth": normalise(
                [p["growth"] for p in selected], True
            ),
            "crime": normalise(
                [p["crime"] for p in selected], False
            ),
            "school": normalise(
                [p["school"] for p in selected], False
            ),
            "hospital": normalise(
                [p["hospital"] for p in selected], False
            ),
            "work": normalise(
                [p["work"] for p in selected], False
            )
        }

        weight = {
            k: numbers[k].value / 100
            for k in keys
        }

        scored = []

        for i, p in enumerate(selected):
            score = sum(
                weight[k] * criteria[k][i]
                for k in keys
            ) * 100

            scored.append((p, score, p["price"] > budget.value))

        scored.sort(key=lambda item: item[1], reverse=True)

        cards = []

        for rank, (p, score, over_budget) in enumerate(scored, 1):
            status = "Over budget" if over_budget else "Within budget"
            colour = "#b42318" if over_budget else "#16803c"

            cards.append(f"""
            <div style="
                border:1px solid #ccc;
                border-radius:10px;
                padding:16px;
                margin:10px 0;
            ">
              <div style="
                  display:flex;
                  justify-content:space-between;
                  flex-wrap:wrap;
                  gap:10px;
              ">
                <div>
                  <div style="color:gray;font-size:12px">RANK #{rank}</div>
                  <div style="font-size:20px;font-weight:bold">{p['postcode']}</div>
                </div>
                <div style="font-size:23px;font-weight:bold">{score:.0f}/100</div>
              </div>
              <hr>
              <div style="display:flex;flex-wrap:wrap;gap:20px">
                <div>Sale price<br><b>£{p['price']:,}</b></div>
                <div>Growth rate<br><b>{p['growth']:.1f}%</b></div>
                <div>Crime index<br><b>{p['crime']}</b></div>
                <div>School distance<br><b>{p['school']:.1f} mi</b></div>
                <div>Hospital distance<br><b>{p['hospital']:.1f} mi</b></div>
                <div>Work distance<br><b>{p['work']:.1f} mi</b></div>
              </div>
              <p style="color:{colour};font-weight:bold">{status}</p>
            </div>
            """)

        display(widgets.HTML("".join(cards)))

        display(widgets.HTML("""
        <p style="font-size:12px;color:gray">
        Demo only: property figures are illustrative, not live market data.
        Entered school, hospital and workplace names are not geocoded yet.
        Scores compare only the selected properties.
        </p>
        """))

compare_button = widgets.Button(
    description="Compare properties",
    button_style="primary",
    icon="search",
    layout=widgets.Layout(width="220px", height="40px")
)
compare_button.on_click(compare_properties)

# =========================
# Final page
# =========================

app = widgets.VBox([
    title,
    widgets.HTML("<h3>Search and location details</h3>"),
    postcodes,
    budget,
    school_name,
    hospital_name,
    workplace_name,
    widgets.HTML("<h3>Your preferences</h3>"),
    bedrooms,
    preferences_box,
    total_label,
    widgets.HTML("<h3>Custom preference (optional)</h3>"),
    widgets.Text(
        description="Preference:",
        placeholder="e.g. Near parks",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="100%", max_width="450px")
    ),
    widgets.HTML(
        "<small style='color:gray'>"
        "Custom preference is not included in the six fixed percentages.</small>"
    ),
    compare_button,
    widgets.HTML("<h3>Preference chart</h3>"),
    chart_output,
    widgets.HTML("<h3>Property comparison</h3>"),
    results_output
], layout=widgets.Layout(
    width="100%",
    max_width="1100px",
    min_width="0px",
    padding="12px",
    gap="8px"
))

refresh_total()
refresh_chart()
display(app)
compare_properties()