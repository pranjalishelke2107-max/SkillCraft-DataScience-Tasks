import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (skip first 4 metadata rows)
df = pd.read_csv("API_SP.POP.TOTL_DS2_en_csv_v2_451027.csv", skiprows=4)

# Display first 5 rows
print(df.head())

# Select required columns
df = df[["Country Name", "2023"]]

# Remove missing values
df = df.dropna()

# Convert population to numeric
df["2023"] = pd.to_numeric(df["2023"])

# Top 10 countries
top10 = df.sort_values(by="2023", ascending=False).head(10)

# Create Bar Chart
plt.figure(figsize=(12,6))
plt.bar(top10["Country Name"], top10["2023"])

plt.title("Top 10 Countries by Population (2023)")
plt.xlabel("Country")
plt.ylabel("Population")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()