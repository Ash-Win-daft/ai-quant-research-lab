
import pandas as pd
from data import validate_market_data

dates = pd.to_datetime([
    "2025-01-02",
    "2025-01-03",
    "2025-01-06"
])

# Test 1: Valid data
good_data = pd.DataFrame(
    {"Close": [100.0, 102.0, 101.0]},
    index=dates
)
print("Test 1:", validate_market_data(good_data))

# Test 2: Missing price
bad_data = good_data.copy()
bad_data.iloc[1, 0] = None

try:
    validate_market_data(bad_data)
except ValueError as error:
    print("Test 2 correctly rejected bad data:", error)

# Test 3: Zero price
bad_price_data = good_data.copy()
bad_price_data.iloc[1, 0] = 0

try:
    validate_market_data(bad_price_data)
except ValueError as error:
    print("Test 3 correctly rejected bad data:", error)

# Test 4: Duplicate dates
duplicate_data = good_data.copy()
duplicate_data.index = pd.to_datetime([
    "2025-01-02",
    "2025-01-02",
    "2025-01-06"
])

try:
    validate_market_data(duplicate_data)
except ValueError as error:
    print("Test 4 correctly rejected duplicate dates:", error)

# Test 5: Dates out of order
unsorted_data = good_data.iloc[::-1]

try:
    validate_market_data(unsorted_data)
except ValueError as error:
    print("Test 5 correctly rejected unsorted dates:", error)
