import numpy as np
import joblib
from assessment import recommend_starting_chapters

# Load the ML model for course recommendation
course_model = joblib.load('models/course_recommender_model.pkl')

def evaluate_assessment_with_ml(scores):
    """
    Enhanced ML-based evaluation for the user assessment.
    
    Args:
        scores (dict): Dictionary with course scores from the assessment
        
    Returns:
        tuple:
            - recommended_course (str): Course recommended by ML model
            - recommended_chapters (dict): Recommended starting chapter for each course
    """
    # Calculate percentage scores assuming 5 questions per course
    percentages = {
        "Python": (scores['Python'] / 5) * 100,
        "Data Analytics": (scores['Data Analytics'] / 5) * 100,
        "Full Stack": (scores['Full Stack'] / 5) * 100
    }

    # Prepare feature vector for the course ML model
    feature_vector = [[
        percentages["Python"],
        percentages["Data Analytics"],
        percentages["Full Stack"]
    ]]

    # Predict the recommended course using the ML model
    recommended_course = course_model.predict(feature_vector)[0]

    # Predict recommended chapters using existing logic (possibly another ML model)
    recommended_chapters = recommend_starting_chapters(percentages)

    return recommended_course, recommended_chapters
