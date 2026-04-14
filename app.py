import argparse
from pathlib import Path

from preprocess import load_consumption_data, build_feature_matrix
from model import train_model, evaluate_model, save_model, load_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "train", "evaluate"])
    args = parser.parse_args()

    data_path = Path("datax/energy_data.csv")
    model_path = Path("models/theft_detector.joblib")

    df = load_consumption_data(data_path)
    X, y, features = build_feature_matrix(df)

    if args.action == "generate":
        print("Dataset ready:", len(df))
        return

    if args.action == "train":
        model = train_model(X, y)
        model_path.parent.mkdir(exist_ok=True)
        save_model(model, model_path)
        print("Model trained & saved")
        return

    if args.action == "evaluate":
        model, _ = load_model(model_path)
        print(evaluate_model(model, X, y))


if __name__ == "__main__":
    main()