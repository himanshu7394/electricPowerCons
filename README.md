# Household Electric Power Consumption Forecasting

This project uses the UCI Individual Household Electric Power Consumption dataset to forecast household energy usage. The work combines unsupervised clustering with supervised time-series modelling. LightGBM is used as the main forecasting model, and hyperparameters are tuned using `RandomizedSearchCV` with time-series cross-validation.

The dataset contains minute-level readings from a household in Sceaux, France (December 2006 to November 2010). It is available here: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

---

## Tech Stack

- **Python**
- **LightGBM**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib**

Additional libraries used in the notebook: Seaborn and CatBoost.

---

## Key Features

- Preprocessing of a large time-series dataset (missing values, resampling, and feature preparation)
- Unsupervised clustering (`MiniBatchKMeans`) to capture usage patterns
- Time-series forecasting with LightGBM
- Time-series cross-validation using `TimeSeriesSplit` (avoids data leakage from future timestamps)
- Hyperparameter tuning with `RandomizedSearchCV`
- Feature importance plots to interpret the model
- Custom preprocessing and modelling steps grouped into a simple pipeline-style workflow

---

## Dataset

| Item | Details |
| --- | --- |
| Source | UCI Machine Learning Repository |
| File | `household_power_consumption.txt` |
| Size | ~2 million minute-level records, 9 variables |
| Target | Household global active power (kW) |

Place the downloaded `.txt` file in the project root before running the notebook.

---

## Project Structure

```text
electricPowerCons/
├── electricPowerCons.ipynb              # main notebook (EDA, clustering, LightGBM)
├── household_power_consumption.txt      # dataset (download from UCI)
├── lightgbm_time_series_model.pkl       # trained LightGBM forecasting model
├── README.md
└── LICENSE
```

---

## How to Run

1. Clone this repository and move into the project folder.

```bash
cd electricPowerCons
```

2. (Recommended) Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required packages.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm catboost jupyter
```

4. Download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and keep `household_power_consumption.txt` in the same folder as the notebook.

5. Open and run the notebook.

```bash
jupyter notebook electricPowerCons.ipynb
```

Run the cells in order. The notebook covers data loading, exploratory analysis, clustering, LightGBM training, hyperparameter search, and evaluation (RMSE, MAE, R²). The trained LightGBM model is saved as `lightgbm_time_series_model.pkl`.

---

## Notes

This is an academic / portfolio Machine Learning project. Results depend on the exact preprocessing, split, and search settings used in the notebook. The MIT License applies to the code in this repository; the dataset remains under its original UCI terms.
