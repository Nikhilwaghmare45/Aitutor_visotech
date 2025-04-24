# course_classifier.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_course_recommender():
    data = []
    for py in range(0, 101, 10):
        for da in range(0, 101, 10):
            for fs in range(0, 101, 10):
                max_score = max(py, da, fs)
                if py == max_score:
                    label = "Python"
                elif da == max_score:
                    label = "Data Analytics"
                else:
                    label = "Full Stack"
                data.append([py, da, fs, label])
    
    df = pd.DataFrame(data, columns=["Python", "Data Analytics", "Full Stack", "Recommended Course"])
    
    X = df[["Python", "Data Analytics", "Full Stack"]]
    y = df["Recommended Course"]
    
    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/course_recommender_model.pkl")

# Run once to generate the model
if __name__ == "__main__":
    train_course_recommender()
