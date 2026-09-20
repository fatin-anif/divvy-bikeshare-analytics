import pandas as pd

df = pd.read_csv("Jan26.csv")

print(df.shape)
print(df.head())

print()
print(df.isna().sum())

print()
print("Duplicate ride_ids:", df["ride_id"].duplicated().sum())

print()
df["started_at"] = pd.to_datetime(df["started_at"])
df["ended_at"] = pd.to_datetime(df["ended_at"])

df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

print(df["duration_min"].describe())

print()
print("Near-zero duration (<1 min):", (df["duration_min"] < 1).sum())
print("Very long duration (>24 hrs):", (df["duration_min"] > 1440).sum())

print()
print(df["rideable_type"].value_counts())
print()
print(df["member_casual"].value_counts())

print()
clean = df[df["duration_min"].between(1, 1440)]
print(clean.groupby("rideable_type")["duration_min"].agg(["mean", "median"]).round(2))

print()
df["hour"] = df["started_at"].dt.hour
hourly_bike = df.groupby(["hour", "rideable_type"]).size().unstack()
print(hourly_bike)

df = df[(df["started_at"] >= "2026-01-01") & (df["started_at"] < "2026-02-01")]


print()
daily = df.groupby(df["started_at"].dt.date).size()
daily_ma7 = daily.rolling(window=7).mean()

print(daily_ma7.tail(10))

print()
df["day_name"] = df["started_at"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_day = df.groupby("day_name").size().reindex(day_order)
print(by_day)

print()
df["week"] = df["started_at"].dt.isocalendar().week
weekly = df.groupby("week").size()
print(weekly)

print()
q1 = df["duration_min"].quantile(0.25)
q3 = df["duration_min"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1 (25th percentile):", round(q1, 2))
print("Q3 (75th percentile):", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower bound:", round(lower_bound, 2))
print("Upper bound:", round(upper_bound, 2))

print()
outliers_iqr = (df["duration_min"] > upper_bound).sum()
print("Rides above IQR upper bound (23.02 min):", outliers_iqr)
print("As % of all rides:", round(outliers_iqr / len(df) * 100, 2), "%")

print()
bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, 120, 99999]
labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-45", "45-60", "60-120", "120+"]
df["duration_bucket"] = pd.cut(df["duration_min"], bins=bins, labels=labels)
print(df["duration_bucket"].value_counts().sort_index())

print()
very_long = df[df["duration_min"] > 120]
print(very_long["duration_min"].describe())
print()
print(very_long["rideable_type"].value_counts())

print()
station_counts = df["start_station_name"].value_counts()
total_stations = len(station_counts)
total_rides_with_station = station_counts.sum()

top10_rides = station_counts.head(10).sum()
top10_pct = top10_rides / total_rides_with_station * 100

print("Total unique start stations:", total_stations)
print("Top 10 stations account for:", round(top10_pct, 1), "% of all station-based rides")

print()
sorted_counts = df["start_station_name"].value_counts().sort_values(ascending=False)
cumulative_pct = (sorted_counts.cumsum() / sorted_counts.sum() * 100)

# sample every 50th station so the output isn't 1139 lines long
print(cumulative_pct.iloc[::50])

print()
sorted_counts = df["start_station_name"].value_counts().sort_values(ascending=False)
total = sorted_counts.sum()

milestones = [10, 20, 50, 100, 200, 300, 500, 1139]
for n in milestones:
    pct = sorted_counts.head(n).sum() / total * 100
    print(f"Top {n} stations: {round(pct, 1)}%")

    print()
df["is_commute_hour"] = df["hour"].isin([7, 8, 16, 17])

station_stats = df.groupby("start_station_name").agg(
    total_rides=("ride_id", "count"),
    pct_commute_hour=("is_commute_hour", "mean"),
    pct_member=("member_casual", lambda x: (x == "member").mean()),
    pct_electric=("rideable_type", lambda x: (x == "electric_bike").mean())
)

# Only look at stations with meaningful ride volume
station_stats = station_stats[station_stats["total_rides"] >= 100]

print(station_stats.shape)
print(station_stats.head())

print()
def classify_station(row):
    if row["pct_commute_hour"] > 0.339 and row["pct_member"] > 0.831:
        return "Commuter Hub"
    elif row["pct_electric"] > 0.610 and row["pct_member"] < 0.831:
        return "Leisure Hub"
    else:
        return "Mixed Use"

station_stats["segment"] = station_stats.apply(classify_station, axis=1)
print(station_stats["segment"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print()
df["day_of_week_num"] = df["started_at"].dt.dayofweek
df["is_electric"] = (df["rideable_type"] == "electric_bike").astype(int)
df["is_member"] = (df["member_casual"] == "member").astype(int)

features = df[["hour", "duration_min", "day_of_week_num", "is_electric"]]
target = df["is_member"]

print(features.head())
print(target.head())

print()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Test set accuracy:", round(accuracy * 100, 2), "%")

print()
baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print("Baseline (always guess majority class):", round(baseline_accuracy * 100, 2), "%")
print("Model accuracy:", round(accuracy * 100, 2), "%")

print()
print("Predicted casual rides:", (predictions == 0).sum())
print("Actual casual rides in test set:", (y_test == 0).sum())

print()
model_balanced = LogisticRegression(max_iter=1000, class_weight="balanced")
model_balanced.fit(X_train, y_train)

predictions_balanced = model_balanced.predict(X_test)
accuracy_balanced = accuracy_score(y_test, predictions_balanced)

print("Balanced model accuracy:", round(accuracy_balanced * 100, 2), "%")
print("Predicted casual rides (balanced):", (predictions_balanced == 0).sum())
print("Actual casual rides in test set:", (y_test == 0).sum())

from sklearn.metrics import confusion_matrix

print()
cm = confusion_matrix(y_test, predictions_balanced)
print("Confusion Matrix:")
print("                 Predicted Casual   Predicted Member")
print("Actual Casual   ", cm[0][0], "            ", cm[0][1])
print("Actual Member   ", cm[1][0], "            ", cm[1][1])

from sklearn.metrics import classification_report

print()
print(classification_report(y_test, predictions_balanced, target_names=["Casual", "Member"]))

import numpy as np

print()
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_valid = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
df_valid["distance_km"] = haversine_distance(
    df_valid["start_lat"], df_valid["start_lng"],
    df_valid["end_lat"], df_valid["end_lng"]
)

print(df_valid["distance_km"].describe())

print()
correlation = df_valid[["distance_km", "duration_min"]].corr()
print(correlation)

print()
non_zero = df_valid[df_valid["distance_km"] > 0]
correlation_nonzero = non_zero[["distance_km", "duration_min"]].corr()
print("Correlation excluding round-trips (0km):")
print(correlation_nonzero)
print()
print("Round-trip rides removed:", len(df_valid) - len(non_zero))

print()
non_zero_clean = non_zero[(non_zero["duration_min"] >= 1) & (non_zero["duration_min"] <= 1440)]

for bike_type in non_zero_clean["rideable_type"].unique():
    subset = non_zero_clean[non_zero_clean["rideable_type"] == bike_type]
    corr = subset[["distance_km", "duration_min"]].corr().iloc[0, 1]
    print(bike_type, "correlation:", round(corr, 3))

    print()
def c_to_f(celsius):
    return celsius * 9/5 + 32

daily_temps_c = {
    1: (-6, -11), 2: (-4, -7), 3: (-2, -6), 4: (1, -4), 5: (7, 0),
    6: (8, 3), 7: (7, 1), 8: (16, 5), 9: (15, 5), 10: (4, -3),
    11: (1, -6), 12: (6, -3), 13: (10, 2), 14: (6, -7), 15: (-4, -9),
    16: (3, -5), 17: (-3, -11), 18: (-7, -13), 19: (-7, -18), 20: (-5, -16),
    21: (1, -4), 22: (-3, -13), 23: (-15, -23), 24: (-13, -22), 25: (-8, -14),
    26: (-13, -17), 27: (-10, -14), 28: (-8, -17), 29: (-8, -17), 30: (-7, -14),
    31: (-3, -11)
}

temp_df = pd.DataFrame([
    {"date_num": day, "avg_temp_f": c_to_f((high + low) / 2)}
    for day, (high, low) in daily_temps_c.items()
])

print(temp_df)

print()
daily_rides = df.groupby(df["started_at"].dt.day).size().reset_index(name="total_rides")
daily_rides = daily_rides.rename(columns={"started_at": "date_num"})

merged = daily_rides.merge(temp_df, on="date_num")
print(merged)

print()
correlation = merged[["avg_temp_f", "total_rides"]].corr().iloc[0, 1]
print("Correlation between temperature and daily rides:", round(correlation, 3))