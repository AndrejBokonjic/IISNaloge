import os
import sys
import joblib
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import DatePreprocessor, SlidingWindowTransformer

params = yaml.safe_load(open("params.yaml"))["train"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]

os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

preprocessed_dir = Path("data/preprocessed/air")
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

Path("models").mkdir(parents=True, exist_ok=True)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

for station in stations:
    print(f"\n--- Treniram model za postajo: {station} ---")

    df = pd.read_csv(f"data/preprocessed/air/{station}.csv")
    df = df[["date_to", target_col]]

    if df[target_col].isna().all():
        print(f"⚠️  Postaja {station} nima podatkov za {target_col}, preskakujem.")
        continue

    try:
        date_preprocessor = DatePreprocessor("date_to")
        df = date_preprocessor.fit_transform(df)
        df = df.drop(columns=["date_to"])

        df_test = df.iloc[-test_size:]
        df_train = df.iloc[:-test_size]

        # Pipeline samo za scaling/imputing
        numeric_transformer = Pipeline([
            ("fillna", SimpleImputer(strategy="mean")),
            ("normalize", MinMaxScaler())
        ])
        preprocess = ColumnTransformer([
            ("numeric_transformer", numeric_transformer, [target_col]),
        ])

        # Fit na train, transform oba
        train_scaled = preprocess.fit_transform(df_train)
        test_scaled = preprocess.transform(df_test)

        # Sliding window ločeno
        sw = SlidingWindowTransformer(window_size)
        X_train, y_train = sw.transform(train_scaled)
        X_test, y_test = sw.transform(test_scaled)

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                  callbacks=[early_stopping], verbose=1)

        scaler = preprocess.transformers_[0][1].named_steps["normalize"]
        y_pred = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        print(f"Test MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")

        model.save(f"models/model_{station}.keras")
        joblib.dump(preprocess, f"models/pipeline_{station}.pkl")
        print(f"✅ Model shranjen: models/model_{station}.keras")

    except Exception as e:
        print(f"⚠️  Napaka za {station}: {e}, preskakujem.")
        continue

print("\n✅ Treniranje zaključeno za vse postaje!")