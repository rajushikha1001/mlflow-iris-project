import mlflow
from sklearn.metrics import accuracy_score
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.attacks.evasion import HopSkipJump
from data import load_data
from sklearn.model_selection import train_test_split

# Must be set for SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def evaluate_model(run_id: str):
    # Load logged model
    model_uri = f"runs:/{run_id}/iris_model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load data
    df = load_data()
    X = df.drop('target', axis=1).values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Basic evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Loaded model accuracy: {accuracy:.4f}")

    # Adversarial robustness check – works perfectly on RandomForest
    classifier = SklearnClassifier(model=model)   # ← THIS LINE ONLY!

    print("Generating adversarial examples using HopSkipJump attack (black-box, tree-compatible)...")
    attack = HopSkipJump(classifier=classifier, targeted=False, max_iter=50)
    X_test_adv = attack.generate(x=X_test)

    y_pred_adv = model.predict(X_test_adv)
    accuracy_adv = accuracy_score(y_test, y_pred_adv)
    print(f"Adversarial accuracy: {accuracy_adv:.4f}")
    if accuracy_adv < 0.9:
        print("WARNING: Model is vulnerable to HopSkipJump attack!")
    else:
        print("Model shows good robustness.")

if __name__ == "__main__":
    run_id = "f74e3e1ee999494aa41440a351822f55"   # ← change if you train again
    evaluate_model(run_id)