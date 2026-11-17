
# ============================================================
# ANIMAL SHELTER OUTCOMES - EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD AND INSPECT DATASET 
# ============================================================

df = pd.read_csv("train.csv.gz")

print("DATASET SHAPE:")
print(df.shape)                      # Prints the number of rows and columns in the dataset

print("FIRST FIVE ROWS:")
print(df.head())                     # Prints the first five rows of the dataset

print("DATASET INFORMATION:")
df.info()                            # Prints a concise summary of the dataset, including column names, non-null counts, and data types

print("DATASET SUMMARY:")
print(df.describe(include="all"))    # Prints summary statistics for all columns, including count, unique values, top values, frequency, mean, standard deviation, min, max, and quartiles


# ============================================================
# 2. MISSING VALUES AND OUTCOMES 
# ============================================================

print("MISSING VALUES:")
print(df.isna().sum().sort_values(ascending=False))    # Prints the count of missing values in each column, sorted from largest to smallest

print("ANIMAL OUTCOMES:")
print(df["OutcomeType"].value_counts())                # Prints the count of each outcome type


# ============================================================
# 3. OUTCOME DISTRIBUTION 
# ============================================================

plt.figure(figsize=(9, 5))                            # Creates the size of the graph

sns.countplot(data=df, x="OutcomeType")               # Counts and plots the outcome categories

plt.title("Distribution of Animal Shelter Outcomes")  # Adds the graph title
plt.xlabel("Outcome Type")                            # Labels the horizontal axis
plt.ylabel("Number of Animals")                       # Labels the vertical axis
plt.xticks(rotation=30)                               # Rotates category labels
plt.tight_layout()                                    # Adjusts spacing

plt.savefig("outcome_distribution.png")               # Saves the graph
plt.show()                                            # Shows or pops up the graph window


# ============================================================
# 4. CAT VS DOG OUTPUTS
# ============================================================

animal_outcomes = pd.crosstab(       
    df["AnimalType"],
    df["OutcomeType"],
    normalize="index"                                 # Calculates proportions within each animal type
) * 100                                               # Converts decimals into percentages

print("OUTCOMES BY ANIMAL TYPE (%):")
print(animal_outcomes.round(2))                       # Prints the outcomes by animal type as percentages, rounded to two decimal places

animal_outcomes.plot(kind="bar")                      # Plots the outcomes by animal type as a bar graph

plt.title("Shelter Outcomes by Animal Type")
plt.xlabel("Animal Type")
plt.ylabel("Percentage of Animals")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outcomes_by_animal.png")
plt.show()


# ============================================================
# 5. ANIMAL AGE OUTPUTS
# ============================================================

print("MOST COMMON ANIMAL AGES:")
print(df["AgeuponOutcome"].value_counts().head(10))            # Displays the 10 most frequent age descriptions

age_counts = df["AgeuponOutcome"].value_counts().head(10)        # Stores those counts for plotting

age_counts.plot(kind="bar")                                     # Plots the most common animal ages at outcome

plt.title("Most Common Animal Ages at Outcome")
plt.xlabel("Age at Outcome")
plt.ylabel("Number of Animals")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("age_distribution.png")
plt.show()

# ============================================================
# 6. MISSING VALUES GRAPH 
# ============================================================

missing_counts = df.isna().sum()                        # Counts the number of missing values in each column

missing_counts = missing_counts[missing_counts > 0]     # Keeps only columns with at least one missing value

missing_counts.plot(kind="bar")                         # Missing values bar graph

plt.title("Missing Values in Animal Shelter Dataset")
plt.xlabel("Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("missing_values.png")
plt.show()
