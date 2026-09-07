"""Funzioni riutilizzabili per esempi di machine learning climatico.

Dataset:
NASA GISTEMP v4, anomalia mensile della temperatura globale land-ocean,
in gradi Celsius rispetto alla climatologia di riferimento 1951-1980.
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


NASA_GISTEMP_CSV = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/"
    "GLB.Ts+dSST.csv"
)


def load_gistemp(url: str = NASA_GISTEMP_CSV) -> pd.DataFrame:
    """Scarica e normalizza le anomalie mensili globali NASA GISTEMP.

    Il file CSV viene individuato a partire dalla riga di intestazione e
    trasformato da formato con un mese per colonna a formato lungo.

    Parameters
    ----------
    url : str, default NASA_GISTEMP_CSV
        URL del file CSV NASA GISTEMP da scaricare.

    Returns
    -------
    pandas.DataFrame
        Serie temporale mensile ordinata per data, con le colonne `year`,
        `month_name`, `month`, `anomaly` e `date`. I valori non numerici
        dell'anomalia vengono esclusi.

    Raises
    ------
    StopIteration
        Se il contenuto scaricato non contiene una riga che inizia con
        ``"Year"``.
    """
    raw = urllib.request.urlopen(url).read().decode(
        "utf-8",
        errors="ignore",
    )

    lines = [line for line in raw.splitlines() if line.strip()]
    header_row = next(
        i for i, line in enumerate(lines)
        if line.startswith("Year")
    )

    table = pd.read_csv(io.StringIO("\n".join(lines[header_row:])))
    table = table.rename(columns={table.columns[0]: "year"})

    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    climate = table[["year", *months]].melt(
        id_vars="year",
        var_name="month_name",
        value_name="anomaly",
    )

    climate["anomaly"] = pd.to_numeric(
        climate["anomaly"],
        errors="coerce",
    )

    climate = climate.dropna(subset=["anomaly"])

    month_map = {
        month_name: month_number
        for month_number, month_name in enumerate(months, start=1)
    }

    climate["month"] = climate["month_name"].map(month_map)
    climate["date"] = pd.to_datetime(
        {
            "year": climate["year"],
            "month": climate["month"],
            "day": 1,
        }
    )

    return climate.sort_values("date").reset_index(drop=True)


def make_features(
    climate: pd.DataFrame,
    n_lags: int = 12,
) -> pd.DataFrame:
    """Costruisce feature temporali, stagionali e autoregressive.

    Il trend è espresso in anni dall'inizio della serie, mentre la
    stagionalità è codificata con seno e coseno del mese. Le prime righe,
    prive di tutti i lag richiesti, vengono rimosse.

    Parameters
    ----------
    climate : pandas.DataFrame
        DataFrame con le colonne `date`, `month` e `anomaly`.
    n_lags : int, default 12
        Numero di anomalie passate da aggiungere come feature autoregressive.

    Returns
    -------
    pandas.DataFrame
        Copia del DataFrame con le colonne `t_years`, `sin_month`,
        `cos_month` e `lag_1` fino a `lag_n_lags`.

    Raises
    ------
    KeyError
        Se mancano una o più colonne necessarie.
    """

    data = climate.copy()
    data["t_years"] = (data["date"] - data["date"].min()).dt.days / 365.2425
    data["sin_month"] = np.sin(2.0 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2.0 * np.pi * data["month"] / 12.0)
    for lag in range(1, n_lags + 1): data[f"lag_{lag}"] = data["anomaly"].shift(lag)

    return data.dropna().reset_index(drop=True)


def fit_inference_model(
    climate: pd.DataFrame,
) -> dict:
    """Stima un modello lineare interpretabile trend + stagionalità.

    Il coefficiente di ``t_years`` è una stima del trend, in °C/anno.
    I residui sono una stima empirica del termine epsilon.

    Parameters
    ----------
    climate : pandas.DataFrame
        DataFrame con la colonna `anomaly` e, direttamente o indirettamente,
        le feature `t_years`, `sin_month` e `cos_month`.

    Returns
    -------
    dict
        Dizionario con il modello adattato (`model`), i dati usati (`data`),
        i nomi delle feature (`features`), i valori stimati (`fitted`), i
        residui (`residuals`), il trend in °C/anno e °C/secolo (`trend_degC_per_year`
        e `trend_degC_per_century`), il coefficiente R² (`r2`) e la deviazione
        standard dei residui (`residual_std_degC`).

    Raises
    ------
    KeyError
        Se mancano le colonne richieste per costruire le feature o per
        adattare il modello.
    """
    needed_features = ["t_years", "sin_month", "cos_month"]

    if not set(needed_features).issubset(climate.columns):
        climate = make_features(climate, n_lags=0)

    X = climate[needed_features]
    y = climate["anomaly"]

    model = LinearRegression().fit(X, y)
    fitted = model.predict(X)
    residuals = y - fitted

    return {
        "model": model,
        "data": climate.copy(),
        "features": needed_features,
        "fitted": fitted,
        "residuals": residuals,
        "trend_degC_per_year": model.coef_[0],
        "trend_degC_per_century": 100.0 * model.coef_[0],
        "r2": model.score(X, y),
        "residual_std_degC": residuals.std(),
    }


def chronological_split(
    climate: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide una serie temporale rispettando l'ordine cronologico.

    Parameters
    ----------
    climate : pandas.DataFrame
        DataFrame già ordinato temporalmente da suddividere.
    train_fraction : float, default 0.80
        Frazione iniziale delle osservazioni da assegnare al training set.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Coppia `(train, test)`: la porzione iniziale e quella finale della
        serie, entrambe come copie indipendenti del DataFrame originale.

    Raises
    ------
    ValueError
        Se `train_fraction` non è strettamente compreso tra 0 e 1.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction deve essere strettamente tra 0 e 1.")

    split_index = int(len(climate) * train_fraction)

    train = climate.iloc[:split_index].copy()
    test = climate.iloc[split_index:].copy()

    return train, test


def fit_prediction_models(
    climate: pd.DataFrame,
    train_fraction: float = 0.80,
    random_state: int = 42,
) -> dict:
    """Addestra modelli predittivi e valuta il periodo più recente.

    Il test set è sempre la parte finale della serie: in questo modo il
    modello non usa osservazioni future per imparare il passato.

    Parameters
    ----------
    climate : pandas.DataFrame
        DataFrame con la colonna `anomaly` e, direttamente o indirettamente,
        le feature temporali e i lag da 1 a 12.
    train_fraction : float, default 0.80
        Frazione iniziale della serie da usare per l'addestramento.
    random_state : int, default 42
        Seme per rendere riproducibile l'addestramento della random forest e
        il calcolo della permutation importance.

    Returns
    -------
    dict
        Dizionario con i dati e le suddivisioni (`data`, `train`, `test`),
        le feature (`features`), i modelli addestrati (`models`), le
        previsioni (`predictions`), le metriche MAE, RMSE e R² (`metrics`) e
        l'importanza delle feature della random forest
        (`random_forest_importance`).

    Raises
    ------
    ValueError
        Se `train_fraction` non definisce una suddivisione valida o se i dati
        non sono sufficienti per addestrare e valutare i modelli.
    KeyError
        Se mancano colonne necessarie per costruire le feature.
    """
    needed = {"t_years", "month", "lag_1", "lag_12"}

    if not needed.issubset(climate.columns):
        climate = make_features(climate, n_lags=12)

    features = [
        "t_years",
        "month",
        *[f"lag_{lag}" for lag in range(1, 13)],
    ]

    train, test = chronological_split(
        climate,
        train_fraction=train_fraction,
    )

    X_train = train[features]
    y_train = train["anomaly"]

    X_test = test[features]
    y_test = test["anomaly"]

    polynomial_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )

    random_forest_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )

    models = {
        "polynomial": polynomial_model,
        "random_forest": random_forest_model,
    }

    predictions = {}
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        scores[name] = {
            "MAE_degC": mean_absolute_error(y_test, y_pred),
            "RMSE_degC": mean_squared_error(y_test, y_pred) ** 0.5,
            "R2": r2_score(y_test, y_pred),
        }

    importance_result = permutation_importance(
        random_forest_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )

    importances = pd.Series(
        importance_result.importances_mean,
        index=features,
        name="permutation_importance",
    ).sort_values(ascending=False)

    return {
        "data": climate.copy(),
        "features": features,
        "train": train,
        "test": test,
        "models": models,
        "predictions": predictions,
        "metrics": pd.DataFrame(scores).T,
        "random_forest_importance": importances,
    }
