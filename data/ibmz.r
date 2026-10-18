install.packages(c("httr", "jsonlite"))
install.packages("writexl")

library(writexl)
library(httr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)

file.edit("~/.Renviron")
api_key <- "16AURWNLZF"


nzchar(api_key)
selected_pstcode <- ""

setwd("C:/Users/had/Downloads/IBMZ/IBMZ")
getwd()

postcodes <- read.csv('uk-postcodes.csv')
postcodes <- unique(na.omit(postcodes))

set.seed(42)

postcodes <- postcodes[sample(nrow(postcodes)), ]
postcodes <- head(postcodes, 5000)

View(postcodes)

# Create dataframe
df <- data.frame(
  postcode = postcodes$full.postcode
)

View(df)

# Two list-columns to store API results
df$growth_data <- vector("list", nrow(df))
df$price_data  <- vector("list", nrow(df))
df$crime_rate <- NA_real_
df$hh_income <- NA_real_
df$council_tax_rate <- NA_real_


request_with_backoff <- function(postcode, api_key, max_retries = 5) {
  
  for (attempt in seq_len(max_retries)) {
    
    response <- GET(
      "https://api.propertydata.co.uk/growth",
      add_headers("X-API-Key" = api_key),
      query = list(postcode = postcode)
    )
    
    if (status_code(response) != 429) {
      return(response)
    }
    
    if (attempt == max_retries) {
      stop("Rate limit persists. Stopping after 5 attempts.")
    }
    
    # Respect the server's Retry-After header
    retry_after <- headers(response)[["retry-after"]]
    
    wait <- 30 * 2^(attempt - 1)
    
    # Add backoff if the server's requested wait is short
    if (wait < 30) {
      wait <- max(wait, 30 * 2^(attempt - 1))
    }
    
    message("HTTP 429. Waiting ", wait, " seconds.")
    Sys.sleep(wait)
  }
}

crime_mapping <- c(
  "Very low crime" = 1,
  "Low crime"      = 2,
  "Average crime"  = 3,
  "High crime"     = 4,
  "Very high crime" = 5
)

council_mapping <- c(
  "Low tax" = 1,
  "High tax" = 2
)

fetch_crime <- function(postcode, api_key, max_retries = 5) {
  
  for (attempt in seq_len(max_retries)) {
    
    response <- GET(
      "https://api.propertydata.co.uk/crime",
      add_headers("X-API-Key" = api_key),
      query = list(postcode = postcode)
    )
    
    if (status_code(response) != 429) {
      return(response)
    }
    
    if (attempt == max_retries) {
      stop("Rate limit persists. Stopping after 5 attempts.")
    }
    
    # Respect the server's Retry-After header
    retry_after <- headers(response)[["retry-after"]]
    
    wait <- 30 * 2^(attempt - 1)
    
    # Add backoff if the server's requested wait is short
    if (wait < 30) {
      wait <- max(wait, 30 * 2^(attempt - 1))
    }
    
    message("HTTP 429. Waiting ", wait, " seconds.")
    Sys.sleep(wait)
  }
}

fetch_household_income <- function(postcode, api_key, max_retries = 5) {
  
  for (attempt in seq_len(max_retries)) {
    
    response <- GET(
      "https://api.propertydata.co.uk/household-income",
      add_headers("X-API-Key" = api_key),
      query = list(postcode = postcode)
    )
    
    if (status_code(response) != 429) {
      return(response)
    }
    
    if (attempt == max_retries) {
      stop("Rate limit persists. Stopping after 5 attempts.")
    }
    
    # Respect the server's Retry-After header
    retry_after <- headers(response)[["retry-after"]]
    
    wait <- 30 * 2^(attempt - 1)
    
    # Add backoff if the server's requested wait is short
    if (wait < 30) {
      wait <- max(wait, 30 * 2^(attempt - 1))
    }
    
    message("HTTP 429. Waiting ", wait, " seconds.")
    Sys.sleep(wait)
  }
}

fetch_prop_tax <- function(postcode, api_key, max_retries = 5) {
  
  for (attempt in seq_len(max_retries)) {
    
    response <- GET(
      "https://api.propertydata.co.uk/council-tax",
      add_headers("X-API-Key" = api_key),
      query = list(postcode = postcode)
    )
    
    if (status_code(response) != 429) {
      return(response)
    }
    
    if (attempt == max_retries) {
      stop("Rate limit persists. Stopping after 5 attempts.")
    }
    
    # Respect the server's Retry-After header
    retry_after <- headers(response)[["retry-after"]]
    
    wait <- 30 * 2^(attempt - 1)
    
    # Add backoff if the server's requested wait is short
    if (wait < 30) {
      wait <- max(wait, 30 * 2^(attempt - 1))
    }
    
    message("HTTP 429. Waiting ", wait, " seconds.")
    Sys.sleep(wait)
  }
}


for (postcode in df$postcode) {
  
  i <- match(postcode, df$postcode)
  while (!is.null(df$price_data[[i]])) {
    next
  }
  
  response_growth <- request_with_backoff(postcode, api_key)
  
  if (status_code(response_growth) != 200) {
    message("HTTP error for ", postcode, ": ",
            status_code(response_growth))
    next
  } else {
    print("response growth good!")
  }
  
  status_code(response_growth)
  headers(response_growth)[["retry-after"]]
  
  crime_response <- fetch_crime(postcode, api_key)
  
  if (status_code(crime_response) == 200) {
    print("response crime good!")
  }
  
  household_inc_response <- fetch_household_income(postcode, api_key)
  
  if (status_code(household_inc_response) == 200) {
    print("response hh_inc good!")
  }
  
  council_tax_response <- fetch_prop_tax(postcode, api_key)
  if (status_code(council_tax_response) == 200) {
    print("response council tax good!")
  }
  
  
  result1 <- content(response_growth, as = "parsed")
  result2 <- content(crime_response, as = "parsed")
  result3 <- content(household_inc_response, as = "parsed")
  result4 <- content(council_tax_response, as = "parsed")
  
  if (is.null(result1) || is.null(result2) || is.null(result3) || is.null(result4)) next
  
  if (identical(result1$status, "success") &&
      identical(result2$status, "success") &&
      identical(result3$status, "success") &&
      identical(result4$status, "success")) {
    
    data <- result1$data
    crime_rate1 = unname(crime_mapping[result2$crime_rating])
    hh_income1 = result3$result$average_household_income
    council_rate = unname(council_mapping[result4$council_rating]) 
    
    prices <- lapply(data, function(x) x[[2]])
    growths <- lapply(data, function(x) x[[3]])
    
    # Store the complete response for this postcode
    df$price_data[[i]] <- prices
    df$growth_data[[i]] <- growths
    df$crime_rate[i] <- crime_rate1
    df$hh_income[i] <- hh_income1
    df$council_tax_rate[i] <- council_rate
    
    
    # Inspect the returned structure
    print(postcode)
    str(data)
    
  } else {
    message("API failed for ", postcode)
    print(result1)
  }
  
  if (i %% 25 == 0) {
    saveRDS(df, "postcode_progress.rds")
    message("Progress saved at row ", i)
  }
  Sys.sleep(1)
  
  saveRDS(df, "postcode_progress.rds")
}

df <- postcode_progress



df$growth_data <- lapply(df$growth_data, function(x) {
  
  if (is.null(x) || length(x) == 0) {
    return(numeric(0))
  }
  
  # Preserve positions of NULL elements
  x <- vapply(x, function(value) {
    if (is.null(value) || length(value) == 0) {
      return("0")
    }
    as.character(value[[1]])
  }, character(1))
  
  x <- gsub("%", "", x)
  
  # Convert to numeric
  suppressWarnings(as.numeric(x))
})

df <- df[!sapply(df$price_data, is.null), ]

df$council_tax_rate <- lapply(df$council_tax_rate, function(x) {
  
  if (is.na(x) || length(x) == 0) {
    return(numeric(1))
  }
  
  # Preserve positions of NULL elements
  x <- vapply(x, function(value) {
    if (is.na(value) || length(value) == 0) {
      return("1")
    }
    as.character(value[[1]])
  }, character(1))
  
  x <- gsub("%", "", x)
  
  # Convert to numeric
  suppressWarnings(as.numeric(x))
})

View(df)

# Create a copy of your dataframe
df_export <- df

# Expand growth_data into separate columns
df_export <- df_export %>%
  unnest_wider(
    growth_data,
    names_sep = "_"
  )

# Expand price_data into separate columns
df_export <- df_export %>%
  unnest_wider(
    price_data,
    names_sep = "_"
  )

View(df_export)

# Export to Excel
write_xlsx(df_export, "property_price_data.xlsx")


# Convert price columns into long format
price_long <- df_export %>%
  pivot_longer(
    cols = starts_with("price_data_"),
    names_to = "year_index",
    values_to = "price"
  ) %>%
  mutate(
    year_index = as.integer(
      sub("price_data_", "", year_index)
    ),
    price = as.numeric(price)
  )