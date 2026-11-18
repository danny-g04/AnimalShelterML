import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("EDA_Help/Animal_Shelter_Project/train.csv.gz")
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isna().sum())

#-------------Converting cols into one hot-------------------#
#new col: 1 is for dog, 0 is for cat
df["is_dog"] = (df["AnimalType"] == "Dog").astype(int)

#the names dont matter, if they have one does
df["has_name"] = df["Name"].notna().astype(int)

#----------------------SexuponOutcome Cleaning---------------------------
df = df.dropna(subset=["SexuponOutcome"])
#sex col contains the strings: female, male, or unknown
df["sex"] = np.where(df["SexuponOutcome"].str.contains("Female"), "female",
             np.where(df["SexuponOutcome"].str.contains("Male"), "male", "unknown"))

#fixed contains: yes, no, unknown
df["fixed"] = np.where(df["SexuponOutcome"].str.contains("Neutered|Spayed"), "yes",
               np.where(df["SexuponOutcome"].str.contains("Intact"), "no", "unknown"))

#------------------------AgeuponOutcome Cleaning---------------------------
df = df.dropna(subset=["AgeuponOutcome"])

parts = df["AgeuponOutcome"].str.split(" ", expand=True)
number = parts[0].astype(int)
unit = parts[1].str.rstrip("s")     

# days in each unit, then multiply
days_per_unit = {"year": 365, "month": 30, "week": 7, "day": 1}
df["age_days"] = number * unit.map(days_per_unit)

#------------------------Breed Cleaning---------------------------
df["is_mix"] = df["Breed"].str.contains("Mix|/").astype(int)
df["breed_main"] = df["Breed"].str.split("/").str[0].str.replace(" Mix", "", regex=False)

#------------------------Color Cleaning---------------------------
df["multi_color"] = df["Color"].str.contains("/").astype(int)
df["color_main"] = df["Color"].str.split("/").str[0]

df = df.drop(columns=["Name", "OutcomeSubtype", "AnimalID", "AnimalType", "DateTime","SexuponOutcome", "AgeuponOutcome", "Breed", "Color"] )
print(df.columns)
print(df.dtypes)

feature_df = df.drop(columns="OutcomeType")
outcome_df = df["OutcomeType"]

#stratify keeps the proportions of the outcomes in the split the same as in the data
input_train, input_test, output_train, output_test = train_test_split(
    feature_df, outcome_df, stratify= outcome_df, test_size=.2, random_state= 1)

