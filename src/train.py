import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data import load_data



# Tell MLflow where the tracking server / database is
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def train_model(max_depth=5, n_estimators=100):
    # Set tracking URI (SQLite backend)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Create or get experiment
    experiment_name = "Iris_Classification"
    experiment_id = mlflow.create_experiment(experiment_name) if not mlflow.get_experiment_by_name(experiment_name) else mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    
    with mlflow.start_run(run_name="Run_with_params"):
        # Load data
        df = load_data()
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        
        # Train model
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log artifact (confusion matrix plot)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, cmap="Blues")
        plt.title("Confusion Matrix")
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)  # Clean up
        
        # Log and register model
        mlflow.sklearn.log_model(model, "iris_model")
        mlflow.register_model("runs:/{}/iris_model".format(mlflow.active_run().info.run_id), "IrisModel")
        
        print(f"Accuracy: {accuracy}")
        print("Model logged and registered.")

if __name__ == "__main__":
    train_model()