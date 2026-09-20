# Divvy Bikeshare Analytics — January 2026

A data analytics portfolio project analyzing Divvy Bikeshare trip data (Chicago) using a structured 9-pattern analytical framework, built hands-on in **Python** and **Power BI**.

## Project Goal

To learn and demonstrate a complete data analytics workflow — from raw data to business insight — by working through nine core analytical patterns on a real, messy, public dataset:

**Foundational:** Data Quality → Description → Diagnostic
**Exploratory:** Comparative → Trend → Distribution
**Advanced:** Segmentation → Predictive → Relationship

## Dataset

- Source: Divvy Bikeshare public trip data
- Period analyzed: January 2026 (~137,764 rides after cleaning)
- Fields: ride ID, bike type, start/end time, start/end station, coordinates, rider type (member/casual)

## Tools Used

- **Python** (pandas, NumPy, scikit-learn) — data cleaning, statistical analysis, machine learning
- **Power BI** (DAX, Power Query) — interactive dashboard, 8 pages
- **VS Code** — Python development environment

## Key Findings

**1. Data quality issues aren't always errors.** ~18-20% of rides were missing station names — initially looked like a data problem, but was fully explained by electric bikes legally docking outside official stations.

**2. A single weather event explains a massive ridership swing.** Jan 23-25 saw ridership collapse ~85% below the monthly average — driven by a historic cold snap (wind chills to -36°F) followed immediately by a major snowstorm. Verified against official NWS records.

**3. Weather's impact isn't a one-off — it's a strong, month-wide relationship.** Correlating daily average temperature against daily ride counts across all of January produced **r = 0.864** — a strong positive correlation, confirming and quantifying the cold-snap finding as a consistent pattern, not an isolated event.

**4. Members commute. Casual riders don't.** Member rides show a textbook 8am/5pm commuter double-peak. Casual riders show almost no morning spike — only 13% of 6-9am rides are casual, vs. 39% overnight — indicating leisure/off-peak usage rather than commuting.

**5. Bike type behaves differently than expected.** Electric bikes show a *stronger* correlation between distance and duration (r=0.48) than classic bikes (r=0.11) — motor assistance likely produces more consistent speeds, while classic bike duration depends heavily on individual rider effort and pace.

**6. Usage is broad-based, not concentrated.** Across 1,139 unique stations, the top 10 account for just 8.3% of all rides — it takes roughly 130-150 stations (~13% of the network) to reach half of all ridership. No single station is a point of failure.

**7. Individual-ride prediction has real limits.** A logistic regression model attempting to predict member vs. casual from ride characteristics (hour, duration, day, bike type) revealed the "accuracy paradox" — 82% accuracy looked good but was identical to simply guessing the majority class every time. After correcting for class imbalance, the honest conclusion: population-level patterns (Finding #4) don't reliably translate into individual-ride predictions with this feature set.

## Dashboard Pages

| Page | Pattern | What it shows |
|---|---|---|
| Overview | Description | KPIs, daily ride trend, top 10 stations |
| Diagnostic | Diagnostic | The Jan 23-25 cold snap/snowstorm collapse |
| Comparative - Member vs Casual | Comparative | Commute-hour usage split by rider type |
| Comparative - Bike Type | Comparative | Usage patterns by classic vs. electric bike |
| Trend - January | Trend | 7-day moving average, day-of-week patterns |
| Distribution | Distribution | Trip duration spread, station concentration curve |
| Segmentation | Segmentation | Rule-based station personas (Commuter/Leisure/Mixed Use) |
| Relationship | Relationship | Distance vs. duration by bike type; temperature vs. ridership |

## Repository Contents
├── dashboard/
│ ├── January_Viz_Divvy.pbix (full interactive Power BI file)
│ └── dashboard_preview.pdf (static export of all 8 pages)
├── python/
│ └── explore_data.py (data cleaning + all statistical analysis)
└── README.md

## Methodology Notes

- All data quality decisions (flagging vs. deleting outliers, calendar-boundary trimming) are documented and applied consistently rather than silently dropping data.
- Statistical outlier detection (IQR method) is explicitly distinguished from data-quality flagging — an important distinction, since "statistically unusual" does not mean "broken data."
- Predictive modeling results are reported honestly, including a model that did *not* outperform a naive baseline once corrected for class imbalance — a deliberate choice to demonstrate sound analytical judgment over inflated results.
- External weather data was sourced from NWS official records and a secondary calendar-based weather source; a data-integrity issue (Celsius/Fahrenheit unit mismatch) was caught and corrected before use.

## About

Built as a self-directed learning project to build hands-on skills in Python and Power BI, following a structured 9-pattern data analytics framework.
