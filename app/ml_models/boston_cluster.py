import pandas as pd
import h3


def frequency_to_severity(count, max_count):
    if count > 0.75 * max_count:
        return 4
    elif count > 0.5 * max_count:
        return 3
    elif count > 0.25 * max_count:
        return 2
    else:
        return 1


def main():
    # Load cleaned dataset
    df = pd.read_csv("../data/cleaned_boston_with_severity.csv")

    # Drop invalid coordinates
    df = df[(df['Latitude'] > 0) & (df['Longitude'] < 0)].copy()

    # Convert coordinates to H3 index (resolution 8 ~ 300–400m hexes)
    df['h3_index'] = df.apply(lambda row: h3.latlng_to_cell(row['Latitude'], row['Longitude'], 9), axis=1)

    # Count crimes per H3 cluster
    cluster_counts = df['h3_index'].value_counts().to_dict()
    max_count = max(cluster_counts.values()) if cluster_counts else 1

    # Assign cluster and severity score
    df['cluster'] = df['h3_index']
    df['cluster_severity'] = df['h3_index'].apply(lambda h: frequency_to_severity(cluster_counts[h], max_count))

    # Save output
    output_path = "../data/clustered_boston_h3.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Clustered Boston data saved to {output_path}")
    print(f"✅ Unique clusters (hexes): {len(cluster_counts)}")


# if __name__ == "__main__":
#     main()
