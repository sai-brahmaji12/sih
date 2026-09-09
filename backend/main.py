from generate_mock_data import generate
from data_processing import clean
from feature_engineering import engineer_features
from classify import train, predict_new
from risk_scoring import compute_risk

def run_pipeline():
    print("=" * 60)
    print("STEP 1-2: Mock Data Generation")
    print("=" * 60)
    raw_df = generate()
    print(f"Generated {len(raw_df)} mock hotspot records")

    print("=" * 60)
    print("STEP 3: Data Processing")
    print("=" * 60)
    cleaned_df = clean(raw_df)

    print("=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)
    feature_df = engineer_features(cleaned_df)

    print("=" * 60)
    print("STEP 5-7: Model Training and Classification")
    print("=" * 60)
    pipe, (X_test, y_test, y_pred) = train(feature_df)

    incoming_df = feature_df.loc[X_test.index].drop(columns=["true_class"])
    classified_df = predict_new(pipe, incoming_df)
    classified_df["true_class"] = feature_df.loc[
        X_test.index, "true_class"
    ]

    print("=" * 60)
    print("STEP 8: Risk Scoring")
    print("=" * 60)
    scored_df = compute_risk(classified_df)
    scored_df.to_csv("final_output.csv", index=False)

    print("\nSaved final_output.csv")
    print("\nRisk level breakdown:")
    print(scored_df["risk_level"].value_counts())

    columns = [
        "hotspot_id", "region", "predicted_class",
        "predicted_confidence", "frp_mw",
        "persistence_count", "dist_to_industry_km",
        "risk_score", "risk_level"
    ]
    print("\nHighest-risk records:")
    print(
        scored_df.sort_values("risk_score", ascending=False)[columns]
        .head(15)
        .to_string(index=False)
    )
    return scored_df

if __name__ == "__main__":
    run_pipeline()
