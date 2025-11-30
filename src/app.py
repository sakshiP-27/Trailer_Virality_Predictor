import streamlit as st
import os
import dotenv
from typing import Optional

# Import prediction logic and data collector methods
from virality_predictor import predict_virality
from data_collector import YouTubeTrailerCollector, extract_video_id


def load_api_key():
    """Load API Key from Streamlit secrets or environment."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except:
        # Fallback to environment variable (for local)
        dotenv.load_dotenv()
        api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        st.error("⚠️ YouTube API Key not found! Add it to Streamlit secrets.")
        st.stop()

    return api_key


def get_virality_category(views_per_hour: float) -> tuple[str, str]:
    """
    Determine virality category based on views per hour.

    Returns:
        tuple: (category_name, emoji)
    """
    if views_per_hour < 100:
        return ("Low Virality", "🔵")
    elif views_per_hour < 1000:
        return ("Moderate Virality", "🟢")
    elif views_per_hour < 10000:
        return ("High Virality", "🟡")
    else:
        return ("Extremely Viral", "🔴")


def main_app():
    st.set_page_config(page_title="Movie Trailer Virality Predictor", layout="wide")
    st.title("🎬 Movie Trailer Virality Predictor")

    api_key = load_api_key()
    collector = YouTubeTrailerCollector(api_key)

    col1, col2 = st.columns([2, 1])

    with col1:
        video_url = st.text_input(
            "Paste YouTube Trailer Link",
            value="https://www.youtube.com/watch?v=kYI2R8V4h-E",  # Example link
            key="video_url_input"
        )

        video_id = extract_video_id(video_url)

        if video_url:
            st.video(video_url)

    with col2:
        st.subheader("Prediction Result")

        if st.button("Predict Virality Score", use_container_width=True):
            if not video_id:
                st.error("Please enter a valid YouTube video URL.")
            elif not os.path.exists("models/virality_model.pkl"):
                st.warning("Model not trained yet. Please run `main.py` first to train the model and save the required files.")
            else:
                with st.spinner(f"Fetching live data for video ID: {video_id}..."):
                    # Step 1: Get live metadata
                    metadata = collector.get_video_metadata_for_prediction(video_id)

                    if not metadata:
                        st.error("Could not fetch video details. Check the URL or API key limits.")
                        return

                    # Step 2: Get live comments
                    comments = collector.get_comments(video_id, max_comments=50)

                    # Step 3: Predict
                    virality_score = predict_virality(metadata, comments)

                    # Calculate views per hour (reverse the scaling)
                    views_per_hour = virality_score / 1000

                    # Get virality category
                    category, emoji = get_virality_category(views_per_hour)

                    # Display result
                    st.metric(
                        label="Virality Score (Scaled Projection)",
                        value=f"{virality_score:,.0f}",
                        delta_color="off"
                    )

                    # Display views per hour
                    st.metric(
                        label="Predicted Views Per Hour",
                        value=f"{views_per_hour:,.1f}",
                        delta_color="off"
                    )

                    # Display category with styling
                    st.markdown(f"""
                        <div style="padding: 15px; border-radius: 10px; background-color: #000000; margin: 10px 0;">
                            <h3 style="margin: 0; text-align: center;">{emoji} {category}</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    # Show the scale
                    st.markdown("---")
                    st.markdown("**Virality Scale:**")
                    st.markdown("""
                        - 🔵 **Low virality:** < 100 views/hour
                        - 🟢 **Moderate virality:** 100 - 1,000 views/hour  
                        - 🟡 **High virality:** 1,000 - 10,000 views/hour
                        - 🔴 **Extremely viral:** > 10,000 views/hour
                    """)

                    st.markdown("---")
                    st.markdown("""
                        <small style="color: #666;">
                        This score is a scaled metric (Views per Hour × 1000). 
                        A higher score indicates a prediction of faster view accumulation.</small>
                    """, unsafe_allow_html=True)

                    st.info(f"**Views**: {metadata['views']:,} | **Likes**: {metadata['likes']:,} | **Comments**: {metadata['number_of_comments']:,}")


if __name__ == "__main__":
    main_app()
