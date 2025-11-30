import requests
import csv
import time
import re
from typing import List, Dict, Optional, Any


def extract_video_id(url: str) -> str:
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


class YouTubeTrailerCollector:
    """Collects YouTube trailer data using YouTube Data API v3"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.quota_used = 0
        self.channel_cache = {}  # Cache channel subscriber counts

    def search_trailers(self, query: str = "official trailer",
                        max_results: int = 500,
                        published_after: Optional[str] = None,
                        published_before: Optional[str] = None) -> List[str]:
        """
        Search for trailer videos and return video IDs

        Args:
            query: Search query (default: "official trailer")
            max_results: Maximum number of results to return
            published_after: ISO 8601 date string (e.g., "2023-01-01T00:00:00Z")
            published_before: ISO 8601 date string (e.g., "2024-12-31T23:59:59Z")

        Returns:
            List of video IDs
        """
        video_ids = []
        next_page_token = None

        print(f"Searching for trailers with query: '{query}'...")

        while len(video_ids) < max_results:
            params = {
                'part': 'id',
                'q': query,
                'type': 'video',
                'maxResults': min(50, max_results - len(video_ids)),
                'key': self.api_key,
                'videoDuration': 'any',
                'relevanceLanguage': 'en'
            }

            if published_after:
                params['publishedAfter'] = published_after

            if published_before:
                params['publishedBefore'] = published_before

            if next_page_token:
                params['pageToken'] = next_page_token

            response = requests.get(f"{self.base_url}/search", params=params)
            self.quota_used += 100  # search.list costs 100 units

            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                break

            data = response.json()

            for item in data.get('items', []):
                if 'videoId' in item['id']:
                    video_ids.append(item['id']['videoId'])

            next_page_token = data.get('nextPageToken')

            print(f"Found {len(video_ids)} videos so far...")

            if not next_page_token:
                break

            time.sleep(0.5)

        print(f"Total videos found: {len(video_ids)}")
        return video_ids

    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for videos

        Args:
            video_ids: List of video IDs

        Returns:
            List of video detail dictionaries
        """
        videos_data = []

        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]

            params = {
                'part': 'snippet,statistics,contentDetails',
                'id': ','.join(batch),
                'key': self.api_key
            }

            response = requests.get(f"{self.base_url}/videos", params=params)
            self.quota_used += 1  # videos.list costs 1 unit

            if response.status_code != 200:
                print(f"Error fetching video details: {response.status_code}")
                continue

            data = response.json()
            videos_data.extend(data.get('items', []))

            print(f"Fetched details for {len(videos_data)} videos...")
            time.sleep(0.5)

        return videos_data

    def get_channel_subscribers(self, channel_id: str) -> int:
        """
        Get subscriber count for a channel

        Args:
            channel_id: YouTube channel ID

        Returns:
            Subscriber count
        """
        if channel_id in self.channel_cache:
            return self.channel_cache[channel_id]

        params = {
            'part': 'statistics',
            'id': channel_id,
            'key': self.api_key
        }

        response = requests.get(f"{self.base_url}/channels", params=params)
        self.quota_used += 1  # channels.list costs 1 unit

        if response.status_code != 200:
            return 0

        data = response.json()
        if data.get('items'):
            sub_count = int(data['items'][0]['statistics'].get('subscriberCount', 0))
            self.channel_cache[channel_id] = sub_count
            return sub_count
        return 0

    def get_comments(self, video_id: str, max_comments: int = 50) -> List[str]:
        """
        Fetch top comments for a video

        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to fetch

        Returns:
            List of comment texts
        """
        comments = []

        params = {
            'part': 'snippet',
            'videoId': video_id,
            'maxResults': min(100, max_comments),
            'order': 'relevance',
            'textFormat': 'plainText',
            'key': self.api_key
        }

        try:
            response = requests.get(f"{self.base_url}/commentThreads", params=params)
            self.quota_used += 1  # commentThreads.list costs 1 unit

            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', [])[:max_comments]:
                    # Extract text from the top-level comment
                    comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment_text)
            elif response.status_code == 403:
                # Comments disabled
                pass
        except Exception as e:
            print(f"Error fetching comments for {video_id}: {str(e)}")

        time.sleep(0.1)  # Reduce sleep time slightly for prediction pipeline
        return comments

    def parse_duration(self, duration: str) -> str:
        """
        Convert ISO 8601 duration to readable format

        Args:
            duration: ISO 8601 duration string (e.g., "PT2M30S")

        Returns:
            Human-readable duration string
        """
        import re

        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)

        if not match:
            return "0:00"

        hours, minutes, seconds = match.groups()
        hours = int(hours) if hours else 0
        minutes = int(minutes) if minutes else 0
        seconds = int(seconds) if seconds else 0

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def get_video_metadata_for_prediction(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and process raw metadata for a single video in the format expected by the predictor.
        """
        videos = self.get_video_details([video_id])
        if not videos:
            return None

        video = videos[0]
        snippet = video['snippet']
        statistics = video.get('statistics', {})
        content_details = video['contentDetails']

        # Get channel subscribers (with caching)
        channel_id = snippet['channelId']
        channel_sub_count = self.get_channel_subscribers(channel_id)

        # Compile data
        video_data = {
            'video_id': video_id,
            'video_title': snippet['title'],
            'description': snippet['description'],
            # Use original key for feature engineering consistency
            'publish_datetime': snippet['publishedAt'],
            'views': int(statistics.get('viewCount', 0)),
            'likes': int(statistics.get('likeCount', 0)),
            'duration': self.parse_duration(content_details['duration']),
            'tags': snippet.get('tags', []),  # Return as list/None for feature_engineer to handle
            'channel_name': snippet['channelTitle'],
            'channel_subscriber_count': channel_sub_count,
            'category_id': snippet.get('categoryId', 'N/A'),
            'number_of_comments': int(statistics.get('commentCount', 0)),
            'thumbnail_url': snippet['thumbnails']['high']['url']
        }
        return video_data

    def collect_dataset(self, num_videos: int = 300,
                        search_query: str = "official trailer",
                        published_after: Optional[str] = None,
                        published_before: Optional[str] = None) -> List[Dict]:
        """
        Main method to collect complete dataset (Original function for training data)
        """
        print("=" * 60)
        print("YouTube Trailer Dataset Collection")
        print("=" * 60)

        # Step 1: Search for trailers
        video_ids = self.search_trailers(search_query, num_videos, published_after, published_before)

        if not video_ids:
            print("No videos found!")
            return []

        # Step 2: Get video details
        print("\nFetching video details...")
        videos = self.get_video_details(video_ids)

        # Step 3: Process each video
        dataset = []

        for idx, video in enumerate(videos, 1):
            print(f"\nProcessing video {idx}/{len(videos)}: {video['snippet']['title'][:50]}...")

            video_id = video['id']
            snippet = video['snippet']
            statistics = video.get('statistics', {})
            content_details = video['contentDetails']

            # Get channel subscribers (with caching)
            channel_id = snippet['channelId']
            channel_sub_count = self.get_channel_subscribers(channel_id)

            # Get comments
            comments = self.get_comments(video_id, 50)

            # Compile data
            video_data = {
                'video_id': video_id,
                'video_title': snippet['title'],
                'description': snippet['description'],
                'publish_datetime': snippet['publishedAt'],
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'duration': self.parse_duration(content_details['duration']),
                'tags': '|'.join(snippet.get('tags', [])),
                'channel_name': snippet['channelTitle'],
                'channel_subscriber_count': channel_sub_count,
                'category_id': snippet.get('categoryId', 'N/A'),
                'number_of_comments': int(statistics.get('commentCount', 0)),
                'comments': '|||'.join(comments),  # Triple pipe separator
                'thumbnail_url': snippet['thumbnails']['high']['url']
            }

            dataset.append(video_data)

            print(f"  Views: {video_data['views']:,} | Likes: {video_data['likes']:,} | Comments: {len(comments)}")
            print(f"  Quota used so far: {self.quota_used}")

        print("\n" + "=" * 60)
        print(f"Collection complete! Total videos: {len(dataset)}")
        print(f"Total quota used: {self.quota_used}")
        print("=" * 60)

        return dataset

    def save_to_csv(self, dataset: List[Dict], filename: str = "youtube_trailers_dataset.csv"):
        """
        Save dataset to CSV file

        Args:
            dataset: List of video data dictionaries
            filename: Output CSV filename
        """
        if not dataset:
            print("No data to save!")
            return

        # Ensure consistent fieldnames by checking keys of the first non-empty dict
        fieldnames = dataset[0].keys() if dataset else []

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)

        print(f"\nDataset saved to: {filename}")
        print(f"Total entries: {len(dataset)}")


# Example usage
if __name__ == "__main__":
    pass
