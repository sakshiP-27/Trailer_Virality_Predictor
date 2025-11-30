import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

# For Feature Engineering
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# For Model Training
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# --- GLOBAL CONFIGURATION ---
PEAK_HOUR_START = 15  # 3 PM UTC
PEAK_HOUR_END = 21    # 9 PM UTC
SCALING_FACTOR = 1000  # For Target Variable calculation


def _extract_video_id(url: str) -> str:
    """Extracts YouTube video ID from a URL."""
    if url is None:
        return ""
    # Standard URL
    match = re.search(r"(?<=v=)[\w-]+", url)
    if match:
        return match.group(0)
    # Shortened URL
    match = re.search(r"youtu\.be\/([\w-]+)", url)
    if match:
        return match.group(1)
    return ""


def _calculate_engagement_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates like ratio and comment ratio, handling division by zero."""
    # Ensure all counts are numeric, default to 0 if missing
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
    df['number_of_comments'] = pd.to_numeric(df['number_of_comments'], errors='coerce').fillna(0)

    # Like Ratio: (Likes / (Views + 1))
    df['like_ratio'] = df['likes'] / (df['views'] + 1)
    # Comment Ratio: (Comments / (Views + 1))
    df['comment_ratio'] = df['number_of_comments'] / (df['views'] + 1)

    return df


def _normalize_tags(tags_value):
    """
    Normalize tags to a consistent string format.
    Handles both list and string inputs.
    """
    if isinstance(tags_value, list):
        return '|'.join(tags_value)
    elif isinstance(tags_value, str):
        return tags_value
    else:
        return ''


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Step 1: Feature Engineering
    Transforms raw dataset features into model-ready features.
    """
    print("Starting Feature Engineering...")
    df = df.copy()

    # --- A. Sentiment Features (VADER for Description and Comments) ---
    analyzer = SentimentIntensityAnalyzer()

    # Description Sentiment (single compound score)
    df['desc_sentiment'] = df['description'].fillna('').apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )

    # Comments Sentiment
    def analyze_comments(comments_str: str) -> Dict[str, float]:
        if not comments_str:
            return {'avg_comment_sentiment': 0.0, 'highly_positive_ratio': 0.0}

        comments = comments_str.split('|||')  # Use the triple pipe separator from data_collector.py
        if not comments:
            return {'avg_comment_sentiment': 0.0, 'highly_positive_ratio': 0.0}

        sentiments = [analyzer.polarity_scores(c.strip())['compound'] for c in comments]
        total_comments = len(sentiments)

        avg_sentiment = np.mean(sentiments)
        # Highly positive: compound score >= 0.5
        highly_positive_count = sum(1 for s in sentiments if s >= 0.5)
        highly_positive_ratio = highly_positive_count / total_comments

        return {
            'avg_comment_sentiment': avg_sentiment,
            'highly_positive_ratio': highly_positive_ratio
        }

    comment_features = df['comments'].fillna('').apply(analyze_comments).apply(pd.Series)
    df = pd.concat([df, comment_features], axis=1)

    # --- B. Time Features ---
    df['publish_datetime'] = pd.to_datetime(df['publish_datetime'])
    df['upload_weekday'] = df['publish_datetime'].dt.weekday  # 0=Monday, 6=Sunday
    df['upload_hour'] = df['publish_datetime'].dt.hour       # 0-23

    # Peak Traffic Hours (e.g., 3 PM to 9 PM UTC)
    df['is_peak_hour'] = df['upload_hour'].apply(
        lambda h: 1 if PEAK_HOUR_START <= h < PEAK_HOUR_END else 0
    )

    # --- D. Text Features ---
    df['title_length'] = df['video_title'].fillna('').str.len()
    df['description_length'] = df['description'].fillna('').str.len()

    # Normalize tags to string format before processing
    df['tags'] = df['tags'].apply(_normalize_tags)
    df['num_tags'] = df['tags'].fillna('').apply(lambda x: len(x.split('|')) if x else 0)

    # --- E. Engagement Features ---
    # Already computed: 'like_ratio', 'comment_ratio' via _calculate_engagement_ratios in main.py prep.
    # Note: views_per_hour requires current time, which is tricky for historical data.
    # We will assume a `hours_since_upload` column is calculated in the main data prep.
    df = _calculate_engagement_ratios(df)  # Recalculate if it wasn't done before loading

    # --- C. Actor Popularity Score (Simplified Mapping) ---
    # Since we cannot use external APIs like pytrends without API keys, we use a simple placeholder.
    # In a real-world scenario, you'd replace this with real data.

    def get_actor_popularity(title: str) -> int:
        title = title.lower()
        if re.search(r"chris evans|tom cruise|leo dicaprio|ryan gosling", title):
            return 90
        elif re.search(r"zendaya|scarlett johansson|chalamet|margot robbie", title):
            return 75
        else:
            return 50

    df['actor_popularity_score'] = df['video_title'].fillna('').apply(get_actor_popularity)

    # --- F. Target Variable (Virality Score) ---
    # We need 'hours_since_upload' for this calculation.
    # Since this function is used for both training (historical) and prediction (live),
    # the caller must ensure 'hours_since_upload' is available.

    df['hours_since_upload'] = (datetime.now(df['publish_datetime'].dt.tz) - df['publish_datetime']).dt.total_seconds() / 3600
    df['views_per_hour'] = df['views'] / (df['hours_since_upload'] + 1e-6)
    df['virality_score'] = df['views_per_hour'] * SCALING_FACTOR

    # Select the final feature set
    final_features = [
        'desc_sentiment',
        'avg_comment_sentiment',
        'highly_positive_ratio',
        'upload_weekday',
        'upload_hour',
        'is_peak_hour',
        'title_length',
        'description_length',
        'num_tags',
        'like_ratio',
        'comment_ratio',
        'actor_popularity_score',
        'views_per_hour'  # Views per hour is an excellent direct proxy
    ]

    return df, final_features


def train_model(data_path: str, model_path: str = "models/virality_model.pkl") -> float:
    """
    Step 2: Model Training
    Loads data, engineers features, trains an XGBoost Regressor, and saves the model.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Cannot train model.")
        return 0.0
    except pd.errors.EmptyDataError:
        print("Error: Dataset is empty. Cannot train model.")
        return 0.0

    # Ensure required columns exist, fill with defaults if they don't (for robustness)
    for col in ['views', 'likes', 'number_of_comments', 'publish_datetime', 'description', 'comments', 'tags']:
        if col not in df.columns:
            df[col] = 0 if col in ['views', 'likes', 'number_of_comments'] else ''

    # 1. Feature Engineering
    df_engineered, feature_names = engineer_features(df)

    # 2. Prepare Data
    X = df_engineered[feature_names]
    y = df_engineered['virality_score']

    # Handle missing values by filling with the mean (standard practice for tree models)
    X = X.fillna(X.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training (XGBoost)
    print("\nTraining XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Training Complete. Evaluation RMSE: {rmse:.2f}")

    # 5. Save Model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save feature names for consistent prediction pipeline
    joblib.dump(feature_names, "models/feature_names.pkl")

    return rmse


def predict_virality(metadata: Dict[str, Any], comments: List[str], model_path: str = "models/virality_model.pkl") -> float:
    """
    Step 3: Prediction Pipeline
    Takes live metadata and comments, engineers features, and predicts the virality score.
    """
    try:
        # Load the trained model and feature list
        model = joblib.load(model_path)
        feature_names = joblib.load("models/feature_names.pkl")
    except FileNotFoundError:
        print(f"Error: Model or feature list not found. Run training first.")
        # Return a neutral score if model is missing
        return 50.0

    # 1. Create a dummy DataFrame for feature engineering
    # Flatten the metadata into a single-row DataFrame
    data_for_df = {
        'video_id': [metadata.get('video_id')],
        'video_title': [metadata.get('video_title')],
        'description': [metadata.get('description')],
        'publish_datetime': [metadata.get('publish_datetime')],
        'views': [metadata.get('views')],
        'likes': [metadata.get('likes')],
        'duration': [metadata.get('duration')],
        'tags': [metadata.get('tags', [])],  # Keep as list, will be normalized in engineer_features
        'channel_name': [metadata.get('channel_name')],
        'channel_subscriber_count': [metadata.get('channel_subscriber_count')],
        'category_id': [metadata.get('category_id')],
        'number_of_comments': [metadata.get('number_of_comments')],
        # Join comments into the format used for training
        'comments': ['|||'.join(comments)],
        'thumbnail_url': [metadata.get('thumbnail_url')]
    }

    df_raw = pd.DataFrame(data_for_df)

    # 2. Feature Engineering on the single row
    df_engineered, _ = engineer_features(df_raw)

    # 3. Prepare features for prediction
    # Ensure features are in the exact order the model expects
    features = df_engineered[feature_names].iloc[0]

    # Handle missing values (should be minimal with the robust feature_engineer)
    features = features.fillna(features.mean() if not features.empty else 0)

    # 4. Prediction
    # XGBoost requires a 2D array, even for a single sample
    prediction = model.predict([features.values])[0]

    # Return the predicted virality score
    return max(0, prediction)  # Score must be non-negative


if __name__ == '__main__':
    # This block is for testing the feature engineering and training functions locally

    # Generate some dummy data to ensure the training function works
    data = {
        'video_id': ['v1', 'v2', 'v3'],
        'video_title': ['Trailer for a New Movie with Tom Cruise and Brad Pitt', 'Zendaya Action Thriller Trailer', 'Independent Short Film Teaser'],
        'description': ['Awesome action movie! You must watch it.', 'A deep psychological drama.', 'Short film debut from a new director.'],
        'publish_datetime': ['2024-10-20T17:00:00Z', '2024-10-21T01:00:00Z', '2024-10-22T23:00:00Z'],
        'views': [1000000, 50000, 5000],
        'likes': [50000, 500, 50],
        'duration': ['PT2M30S', 'PT1M30S', 'PT0M50S'],
        'tags': ['action|trailer|tom cruise', 'drama|zendaya|thriller', 'short|indie'],
        'channel_name': ['Studio A', 'Studio B', 'Indie Creator'],
        'channel_subscriber_count': [10000000, 500000, 5000],
        'category_id': [24, 24, 24],
        'number_of_comments': [1000, 50, 5],
        'comments': [
            'Great trailer!|||Amazing effects!|||Can\'t wait to see this!',
            'Looks good.',
            'Nice.'
        ],
        'thumbnail_url': ['u1', 'u2', 'u3']
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.to_csv("youtube_trailers_dataset.csv", index=False)

    # Ensure models directory exists for saving
    import os
    os.makedirs("models", exist_ok=True)

    # Test the training function (using the dummy data)
    print("-" * 50)
    print("Testing Training Function:")
    train_model("youtube_trailers_dataset.csv")

    # Test the prediction function
    print("-" * 50)
    print("Testing Prediction Function:")

    # Simulate live data for prediction (using v1 data)
    live_metadata = {
        'video_id': 'live_v1',
        'video_title': 'New Action Trailer with Tom Cruise - Official',
        'description': 'The most anticipated movie of the year!',
        'publish_datetime': datetime.now().isoformat(),  # Use current time for "fresh" data
        'views': 1000,
        'likes': 100,
        'duration': 'PT2M0S',
        'tags': ['action', 'trailer', 'tom cruise'],
        'channel_name': 'New Studio'
    }
