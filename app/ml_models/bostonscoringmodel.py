import pandas as pd
import numpy as np
import h3
import os
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def train_model(X, y, model_path="model/xgb_model.pkl"):

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    print("\nSplitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, test_predictions)
    print(f"\nTest Set RMSE: {rmse:.4f}")

    # Print feature importances
    importance_df = pd.DataFrame({
        'feature': ["center"] + [f"n{i+1}" for i in range(6)],
        'importance': model.feature_importances_
    })
    print("\nFeature Importances:")
    print(importance_df.sort_values('importance', ascending=False))

    return model, X_train, X_test, y_train, y_test


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[(df['Latitude'] > 0) & (df['Longitude'] < 0)]
    df = df[['Latitude', 'Longitude', 'cluster', 'cluster_severity']]

    # print("\nOriginal Data Statistics:")
    # print(
    #     f"Severity range before normalization: {df['cluster_severity'].min():.2f} - {df['cluster_severity'].max():.2f}")

    # Direct linear mapping from severity to safety score
    # severity 4 (highest) -> safety 1 (least safe)
    # severity 1 (lowest) -> safety 10 (safest)
    df['safety_score'] = 11 - (df['cluster_severity'] * 2.25)

    # Ensure values are within 1-10 range
    df['safety_score'] = df['safety_score'].clip(1, 10)

    # print(f"Safety scores after conversion: {df['safety_score'].min():.2f} - {df['safety_score'].max():.2f}")
    # print("\nSafety Score Distribution:")
    # print(df['safety_score'].value_counts().sort_index())
    # print("\nSample of safety scores:")
    # print(df[['Latitude', 'Longitude', 'cluster_severity', 'safety_score']].head())

    df.drop_duplicates('cluster', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_neighbor_severities(h3_index, cluster_map, debug=True):
    neighbors = h3.grid_disk(h3_index, 1)
    if h3_index in neighbors:
        neighbors.remove(h3_index)

    median_safety = np.median(list(cluster_map.values()))
    neighbor_scores = [cluster_map.get(n, median_safety) for n in neighbors]

    if debug:
        print(f"Number of neighbors found: {len(neighbors)}")
        print(f"Neighbor cells: {neighbors}")
        print(f"Neighbor safety scores: {neighbor_scores}")

    return neighbor_scores


def create_features(df, cluster_map):
    features = []
    print("\nCreating features:")
    print(f"Total clusters to process: {len(df['cluster'])}")

    for idx, h3_index in enumerate(df['cluster']):
        if idx < 5:  # Print first 5 examples
            print(f"\nProcessing cluster {idx + 1}:")
            print(f"H3 index: {h3_index}")
            print(f"Safety score: {cluster_map[h3_index]:.2f}")

        center = cluster_map[h3_index]
        neighbors = get_neighbor_severities(h3_index, cluster_map, debug=(idx < 5))
        row = [center] + neighbors[:6]

        median_safety = np.median(list(cluster_map.values()))
        while len(row) < 7:
            row.append(median_safety)

        features.append(row)

    feature_df = pd.DataFrame(features, columns=["center"] + [f"n{i + 1}" for i in range(6)])
    print("\nFeature Statistics:")
    print(feature_df.describe())
    return feature_df


def predict_safety_score(lat, lon, model, cluster_map, debug=True):
    h3_index = h3.latlng_to_cell(lat, lon, 9)

    if debug:
        print(f"\nPredicting for coordinates: {lat}, {lon}")
        print(f"H3 Index: {h3_index}")

    # Get center score
    if h3_index in cluster_map:
        center_score = cluster_map[h3_index]
        if debug:
            print(f"Exact cluster match found! Score: {center_score:.2f}")
    else:
        # If no exact match, use median as fallback
        center_score = np.median(list(cluster_map.values()))
        if debug:
            print(f"No exact cluster match found. Using median score: {center_score:.2f}")

    # Get neighbor scores
    neighbors = h3.grid_disk(h3_index, 1)
    if h3_index in neighbors:
        neighbors.remove(h3_index)
    neighbor_scores = [cluster_map.get(n, np.median(list(cluster_map.values()))) for n in neighbors]

    # Combine center and neighbor scores
    all_scores = [center_score] + neighbor_scores[:6]
    while len(all_scores) < 7:
        all_scores.append(np.median(list(cluster_map.values())))

    if debug:
        print(f"All scores (center + 6 neighbors): {all_scores}")

    # Use the minimum score as the final safety score (most conservative)
    final_score = max(1, min(10, round(min(all_scores))))
    confidence = 1.0  # Rule-based, so always 100%

    if debug:
        print(f"Final safety score (min of center+neighbors): {final_score}")
        print(f"Prediction Confidence: {confidence:.2%}")

    return final_score, confidence


def main():
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "clustered_boston_h3.csv")
    model_path = os.path.join(base_dir, "model", "xgb_model.pkl")

    df = load_and_preprocess_data(file_path)
    cluster_map = dict(zip(df['cluster'], df['safety_score']))

    print("\nCluster map statistics:")
    safety_values = list(cluster_map.values())
    print(f"Number of clusters: {len(cluster_map)}")
    print(f"Safety score range: {min(safety_values):.2f} - {max(safety_values):.2f}")
    print(f"Median safety score: {np.median(safety_values):.2f}")

    # Create features and train model
    X = create_features(df, cluster_map)
    y = df['safety_score'].values

    # print("\nTraining model...")
    # model, X_train, X_test, y_train, y_test = train_model(X, y)

    if os.path.exists(model_path):
        print(f"\nLoading pre-trained model from {model_path}")
        model = joblib.load(model_path)
        X_train, X_test, y_train, y_test = None, None, None, None
    else:
        print("\nModel not found. Training a new one...")
        model, X_train, X_test, y_train, y_test = train_model(X, y, model_path=model_path)

    # Prediction testing
    test_coordinates = [
        (42.290177, -71.128854),  # West Roxbury
        (42.358162, -71.059726),  # Downtown Crossing
        (42.352367, -71.061850),  # Theater District
        (42.33721, -71.073505),   # Roxbury
        (42.355923, -71.055582),  # Downtown
        (42.363725, -71.053817),  # Beacon Hill
        (42.338229, -71.084100),  # Back Bay
        (42.311557, -71.053706),  # Dorchester
        (42.329608, -71.084318),  # Nubian Street
        (42.333648, -71.076396),
        (42.343979, -71.044453),
        (42.351035, -71.133987),
        (42.345975, -71.084612),
        (42.359581, -71.050810),
        (42.371859, -71.039602),
        (42.360001, -71.058559),
        (42.377223, -71.056916),
        (42.346641, -71.045912),
        (42.636854, -69.926293)
    ]

    print("\nTesting predictions:")
    for lat, lon in test_coordinates:
        print(f"\n{'=' * 50}")
        score, confidence = predict_safety_score(lat, lon, model, cluster_map)
        print(f"Location: {lat}, {lon}")
        print(f"Final Safety Score (1=least safe, 10=safest): {score}")
        print(f"Prediction Confidence: {confidence:.2f}")
# [Rest of your code remains the same]

# if __name__ == "__main__":
#     main()