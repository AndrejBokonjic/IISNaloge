import os
import sys
import joblib
import random
from pathlib import Path

import yaml
import numpy as np
import mlflow
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
import tf2onnx

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import DatePreprocessor, SlidingWindowTransformer

params = yaml.safe_load(open("params.yaml"))["train"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]

# MLflow init — zamenjaj z svojim DAGsHub URL
mlflow.set_tracking_uri("https://dagshub.com/AndrejBokonjic/IISNaloge.mlflow")

os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

preprocessed_dir = Path("data/preprocessed/air")
stations = [f.stem for f in preprocessed_dir.glob("*.csv")]

Path("models").mkdir(parents=True, exist_ok=True)

def build_model(input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

mlflow.set_experiment("IISNaloge_train")

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

        numeric_transformer = Pipeline([
            ("fillna", SimpleImputer(strategy="mean")),
            ("normalize", MinMaxScaler())
        ])
        preprocess = ColumnTransformer([
            ("numeric_transformer", numeric_transformer, [target_col]),
        ])

        train_scaled = preprocess.fit_transform(df_train)
        test_scaled = preprocess.transform(df_test)

        sw = SlidingWindowTransformer(window_size)
        X_train, y_train = sw.transform(train_scaled)
        X_test, y_test = sw.transform(test_scaled)

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        with mlflow.start_run(run_name=f"train_{station}"):
            mlflow.log_param("station", station)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("window_size", window_size)
            mlflow.log_param("target_col", target_col)
            mlflow.log_param("random_state", random_state)

            model = build_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

            scaler = preprocess.transformers_[0][1].named_steps["normalize"]
            y_pred = model.predict(X_test)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)

            mse = mean_squared_error(y_test_inv, y_pred_inv)
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            print(f"Test MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_rmse", rmse)

            model_path = f"models/model_{station}.keras"
            model.save(model_path)
            mlflow.log_artifact(model_path)

            onnx_path = f"models/model_{station}.onnx"
            tf2onnx.convert.from_keras(model, output_path=onnx_path)
            mlflow.log_artifact(onnx_path)
            print(f"✅ ONNX model shranjen: {onnx_path}")

            pipeline_path = f"models/pipeline_{station}.pkl"
            joblib.dump(preprocess, pipeline_path)
            mlflow.log_artifact(pipeline_path)

            print(f"✅ Model shranjen: {model_path}")
            mlflow.end_run(status="FINISHED")

    except Exception as e:
        print(f"⚠️  Napaka za {station}: {e}, preskakujem.")
        continue

print("\n✅ Treniranje zaključeno za vse postaje!")