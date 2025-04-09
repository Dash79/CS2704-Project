"""
======================================================
IMDB Movie Ratings Analysis
======================================================
Author: Dawit Ashenafi Getachew - 3752264
        Chizaram Ikpo - 3760059
Description: 
This script loads the IMDB dataset, cleans it, and performs 
basic exploratory data analysis (EDA) to gain insights 
into movie ratings, genres, and revenue.
======================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr  # or other statistical tests if needed

# Load Dataset
df = pd.read_csv("IMDB-Movie-Data.csv")

# Check for Missing Values
print("Missing Values:\n", df.isnull().sum())

# Drop Missing Values in Key Column(s)
# Since "Budget (Millions)" doesn't exist, we'll only drop rows missing "Rating".
df.dropna(subset=["Rating"], inplace=True)

# Filter Relevant Columns
df = df[[
    "Rank",
    "Title",
    "Year",
    "Genre",
    "Rating",
    "Revenue (Millions)",
    "Director",
    "Actors"
]]

# Save Cleaned Data
df.to_csv("Cleaned_IMDB_Movie_Data.csv", index=False)
print(" Cleaned dataset saved.")

# Exploratory Data Analysis (EDA)

# Highest-Rated Movie Genres
plt.figure(figsize=(10,5))
genre_ratings = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
sns.barplot(x=genre_ratings.index, y=genre_ratings.values)
plt.xticks(rotation=90)
plt.title("Average IMDB Ratings by Genre")
plt.xlabel("Genre")
plt.ylabel("IMDB Rating")
plt.savefig("ratings_by_genre.png")  # Save the plot as an image
plt.show()

# Ratings Over Time (if 'Year' is valid)
plt.figure(figsize=(10,5))
yearly_ratings = df.groupby("Year")["Rating"].mean().sort_index()
sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values, marker="o")
plt.title("Average IMDB Ratings Over the Years")
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
plt.grid(True)
plt.savefig("ratings_over_time.png")
plt.show()

# OPTIONAL: Revenue vs. Rating Correlation (since you have "Revenue (Millions)")
if "Revenue (Millions)" in df.columns:
    # Drop rows missing Revenue, if you want a correlation
    df.dropna(subset=["Revenue (Millions)"], inplace=True)

    # Compute correlation between Revenue & Rating
    corr, p_value = spearmanr(df["Revenue (Millions)"], df["Rating"])
    print("Spearman Correlation Between Revenue & Rating:", corr)
    print("P-value:", p_value)

    # Plot
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["Revenue (Millions)"], y=df["Rating"], alpha=0.7)
    plt.title("Revenue vs. Rating")
    plt.xlabel("Revenue (Millions)")
    plt.ylabel("IMDB Rating")
    plt.savefig("revenue_vs_rating.png")
    plt.show()

# Save Final Dataset
df.to_csv("Final_IMDB_Analysis.csv", index=False)
print(" Final dataset saved.")
