
#calculate preferences percentage
def process_preferences(preferences):
    if not preferences:
        raise ValueError("Select at least one preference")
            


    total = sum(preferences.values())

    if total != 100:
        raise ValueError("Percentages must be 100")

    weights = {}

    for key, percentage in preferences, items():
        weights[key] = percentage / 100

    return weights
    
#filter by price
def price_filter(properties, min_pri, max_pri):
    if min_pri < 0 or max_pri < min_pri:
        raise ValueError("Invalid price range")

    filtered_properties0 = []

    for property_data in properties:
        price = property_data.get("price")

        if price is None:
            continue

        if min_pri <= price <= max_pri:
            filtered_properties0.append(property_data)

    return filtered_properties0

#filter by bedrooms
def bedroom_filter(filtered_properties0, min_bed_num):
    filtered_properties1 = []
    for property_data in filtered_properties0:
        bed_num = property_data.get("bed_num")
        if bed_num is None:
            continue
        if min_bed_num <= bed_num:
            filtered_properties1.append(property_data)

    return filtered_properties1

#combining filters
def filter_properties(properties, min_price, max_price, min_bed_num, location):
    result1 = price_filter(properties, min_price, max_price)
    result2 = bedroom_filter(result0, min_bed_num)

    return result1

#