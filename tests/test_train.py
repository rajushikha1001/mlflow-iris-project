from src.train import train_model

def test_train_model():
    train_model(max_depth=3, n_estimators=50)
    # Add assertions if needed, e.g., check if files exist
    assert True  # Placeholder for successful run