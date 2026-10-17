import ipywidgets as widgets
from IPython.display import display, clear_output
from plotly.graph_objects import Figure, Pie

# =====================================================
# 1. Preference configuration
# =====================================================

FEATURES = [
    "Price",
    "Growth Rate",
    "Distance to Hospital",
    "Distance to School",
    "Distance to Workplace",
    "Crime Rate"
]

COLORS = [
    "#3B82F6",  # Blue
    "#22C55E",  # Green
    "#F59E0B",  # Orange
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#06B6D4"   # Cyan
]

# Initial equal allocation: 100% total
initial_values = [17, 17, 17, 17, 16, 16]

preferences = dict(zip(FEATURES, initial_values))
confirmed = {name: False for name in FEATURES}

# Prevent recursive callbacks when sliders update each other
updating = False

# =====================================================
# 2. Create sliders, Confirm buttons and budget input
# =====================================================

sliders = {}
confirm_buttons = {}
rows = []

budget_input = widgets.Text(
    value="",
    placeholder="Optional, e.g. 300000",
    description="Budget (£):",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="300px")
)

budget_note = widgets.HTML(
    "<span style='color:#64748b;font-size:12px'>"
    "Optional: leave blank if you do not want a budget limit."
    "</span>"
)

status = widgets.HTML()
chart_output = widgets.Output()
summary_output = widgets.Output()

for i, name in enumerate(FEATURES):

    slider = widgets.IntSlider(
        value=initial_values[i],
        min=0,
        max=100,
        step=1,
        description="",
        readout=True,
        continuous_update=False,
        layout=widgets.Layout(width="300px", flex="1 1 200px")
    )

    percentage = widgets.HTML(
        value=f"<b>{initial_values[i]}%</b>",
        layout=widgets.Layout(width="48px")
    )

    button = widgets.Button(
        description="Confirm",
        button_style="",
        icon="check",
        layout=widgets.Layout(width="110px")
    )

    sliders[name] = slider
    confirm_buttons[name] = button

    label = widgets.HTML(
        value=(
            f"<span style='color:{COLORS[i]};font-size:17px'>●</span> "
            f"<b>{name}</b>"
        ),
        layout=widgets.Layout(width="230px")
    )

    controls = [label, slider, percentage, button]

    if name == "Price":
        controls.append(
            widgets.VBox([budget_input, budget_note])
        )

    row = widgets.HBox(
        controls,
        layout=widgets.Layout(
            width="100%",
            align_items="center",
            flex_flow="row wrap",
            margin="8px 0"
        )
    )

    rows.append(row)

    # Store the percentage widget for live updates
    slider._percentage_label = percentage

def make_percentage_callback(slider_widget):

    def update_percentage(change):
        slider_widget._percentage_label.value = (
            f"<b>{change['new']}%</b>"
        )

    return update_percentage

slider.observe(
    make_percentage_callback(slider),
    names="value"
)

# =====================================================
# 3. Helper: distribute integer percentages exactly
# =====================================================

def distribute_integer(total, weights):
    """Allocate integer percentages that sum exactly to total."""
    if not weights:
        return []

    if total <= 0:
        return [0] * len(weights)

    weight_sum = sum(weights)

    if weight_sum == 0:
        weights = [1] * len(weights)
        weight_sum = len(weights)

    exact = [total * w / weight_sum for w in weights]
    result = [int(x) for x in exact]
    remainder = total - sum(result)

    # Largest remainder method avoids rounding errors
    order = sorted(
        range(len(exact)),
        key=lambda i: exact[i] - result[i],
        reverse=True
    )

    for i in order[:remainder]:
        result[i] += 1

    return result


# =====================================================
# 4. Live preference donut chart
# =====================================================

def render_chart():
    values = [preferences[name] for name in FEATURES]

    with chart_output:
        clear_output(wait=True)

        fig = Figure(
            data=[
                Pie(
                    labels=FEATURES,
                    values=values,
                    hole=0.58,
                    marker=dict(
                        colors=COLORS,
                        line=dict(color="white", width=3)
                    ),
                    textinfo="label+percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "Preference: %{value}%"
                        "<extra></extra>"
                    ),
                    sort=False
                )
            ]
        )

        fig.update_layout(
            title="Your Property Preference Breakdown",
            height=500,
            margin=dict(t=65, b=25, l=20, r=20),
            showlegend=True,
            annotations=[
                dict(
                    text=f"{sum(values)}%<br><sup>Total</sup>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=22)
                )
            ]
        )

        display(fig)


# =====================================================
# 5. Update totals and progress
# =====================================================

def update_status():
    total = sum(preferences.values())
    count = sum(confirmed.values())

    status.value = f"""
    <div style="
        padding:12px;
        border-radius:10px;
        background:#f0fdf4;
        margin:8px 0 16px 0;
    ">
        <b>Total preference: {total}%</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Confirmed: {count}/6
        <div style="
            margin-top:8px;
            background:#d1d5db;
            height:8px;
            border-radius:8px;
        ">
            <div style="
                width:{count / 6 * 100}%;
                height:8px;
                background:#16a34a;
                border-radius:8px;
            "></div>
        </div>
    </div>
    """


def refresh_display():
    for name in FEATURES:
        sliders[name]._percentage_label.value = (
            f"<b>{preferences[name]}%</b>"
        )

        if confirmed[name]:
            confirm_buttons[name].description = "Confirmed"
            confirm_buttons[name].button_style = "success"
            confirm_buttons[name].icon = "lock"

    update_status()
    render_chart()


# =====================================================
# 6. Synchronise all unconfirmed sliders
# =====================================================

def on_slider_change(changed_name, change):
    global updating

    if updating or confirmed[changed_name]:
        return

    updating = True

    try:
        # Read the user's newly selected percentage
        new_value = int(change["new"])
        locked_total = sum(
            preferences[name]
            for name in FEATURES
            if confirmed[name]
        )

        # Never allow confirmed percentages to change
        available = 100 - locked_total
        new_value = max(0, min(new_value, available))

        preferences[changed_name] = new_value

        # Reallocate the remaining percentage across
        # all other unconfirmed features.
        remaining_names = [
            name for name in FEATURES
            if not confirmed[name] and name != changed_name
        ]

        remaining_total = 100 - locked_total - new_value

        old_weights = [
            preferences[name] for name in remaining_names
        ]

        allocations = distribute_integer(
            remaining_total,
            old_weights
        )

        for name, value in zip(remaining_names, allocations):
            preferences[name] = value

        # Update all slider values without recursion
        locked_total = sum(
            preferences[name]
            for name in FEATURES
            if confirmed[name]
        )

        max_unlocked = 100 - locked_total

        for name in FEATURES:
            slider = sliders[name]

            if confirmed[name]:
                continue

            slider.max = max_unlocked
            slider.value = preferences[name]

    finally:
        updating = False

    refresh_display()


for name in FEATURES:
    sliders[name].observe(
        lambda change, feature=name:
            on_slider_change(feature, change),
        names="value"
    )


# =====================================================
# 7. Confirm each preference and lock its slider
# =====================================================

def confirm_preference(name):
    global updating

    if confirmed[name]:
        return

    # Budget is optional, but if entered it must be valid
    if name == "Price":
        raw_budget = budget_input.value.strip()

        if raw_budget:
            try:
                budget = int(raw_budget.replace(",", ""))
                if budget <= 0:
                    raise ValueError
            except ValueError:
                status.value = """
                <div style="color:#dc2626;padding:8px">
                    Please enter a valid positive budget, or leave it blank.
                </div>
                """
                return

    confirmed[name] = True

    # Freeze this slider permanently
    sliders[name].disabled = True
    confirm_buttons[name].disabled = True

    updating = True

    try:
        locked_total = sum(
            preferences[f]
            for f in FEATURES
            if confirmed[f]
        )

        max_unlocked = 100 - locked_total

        for feature in FEATURES:
            if not confirmed[feature]:
                sliders[feature].max = max_unlocked

        # If only one feature remains, its value is forced
        # by the 100% total and cannot be adjusted freely.
        unlocked = [
            f for f in FEATURES if not confirmed[f]
        ]

        if len(unlocked) == 1:
            last = unlocked[0]
            forced_value = 100 - locked_total
            preferences[last] = forced_value
            sliders[last].value = forced_value
            sliders[last].min = forced_value
            sliders[last].max = forced_value

    finally:
        updating = False

    refresh_display()

    if all(confirmed.values()):
        show_final_result()


for name in FEATURES:
    confirm_buttons[name].on_click(
        lambda button, feature=name:
            confirm_preference(feature)
    )


# =====================================================
# 8. Final personalised result
# =====================================================

def show_final_result():
    with summary_output:
        clear_output(wait=True)

        print("All six preferences confirmed!")
        print("=" * 38)

        for name in FEATURES:
            print(f"{name}: {preferences[name]}%")

        print("\nTotal:", sum(preferences.values()), "%")

        raw_budget = budget_input.value.strip()

        if raw_budget:
            budget = int(raw_budget.replace(",", ""))
            print(f"Maximum budget: £{budget:,}")
        else:
            print("Maximum budget: No budget limit specified")

        print("\nYour personalised preference profile is ready.")


# =====================================================
# 9. Render the interface
# =====================================================

title = widgets.HTML("""
<div style="margin-bottom:12px">
    <h2 style="margin-bottom:4px">Property Preferences</h2>
    <p style="color:#64748b">
        Allocate 100% across six factors.
        Adjust a slider, then confirm to lock it.
    </p>
</div>
""")

display(
    widgets.VBox(
        [
            title,
            status,
            *rows,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Personal Preference Breakdown</h3>"),
            chart_output,
            summary_output
        ],
        layout=widgets.Layout(width="100%")
    )
)

refresh_display()






