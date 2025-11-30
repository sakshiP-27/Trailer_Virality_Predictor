import os
import json
import dotenv
from datetime import datetime

# Import the collector and the predictor logic
from src.data_collector import YouTubeTrailerCollector
from src.virality_predictor import train_model


def module_collect_and_load(API_KEY: str):
    # Initialize collector
    collector = YouTubeTrailerCollector(API_KEY)
    all_datasets: list = []

    # Target CSV filename
    DATA_FILE = "youtube_trailers_dataset.csv"

    search_queries: list = [
        "official movie trailer",
        "bollywood official trailer",
        "hollywood official trailer",
        "film trailer official",
        "official movie teaser"
    ]

    # We aim for ~166 + more, so let's set a target collection goal.
    # The user mentioned having 166 already, so we collect 300 more to exceed 400 total.
    NUM_VIDEOS_PER_QUERY: int = 300 // len(search_queries)

    for query in search_queries:
        print(f"Searching for query: {query}")
        dataset = collector.collect_dataset(
            search_query=query,  # Use the specific query here
            num_videos=NUM_VIDEOS_PER_QUERY,
            published_after="2023-01-01T00:00:00Z",
            published_before="2024-12-31T23:59:59Z"
        )
        all_datasets.extend(dataset)

    # Take only unique videos from the dataset
    unique_videos: dict = {video['video_id']: video for video in all_datasets}
    final_dataset: list = list(unique_videos.values())

    # Save to CSV
    collector.save_to_csv(final_dataset, DATA_FILE)

    # Save metadata as JSON for reference
    with open("collection_metadata.json", "w") as f:
        json.dump({
            "collection_date": datetime.now().isoformat(),
            "total_videos": len(final_dataset),
            "quota_used": collector.quota_used,
            "search_query": search_queries
        }, f, indent=2)

    # --- Step 2: Model Training ---
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    rmse = train_model(data_path=DATA_FILE)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETED with RMSE: {rmse:.2f}")
    print("=" * 60)


def main():
    dotenv.load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")

    module_collect_and_load(API_KEY)


if __name__ == "__main__":
    main()
