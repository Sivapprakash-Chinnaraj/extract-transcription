import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up YouTube Data API
API_KEY = "AIzaSyAe9nQVoc9KuZbljckF-n-tpNjQ2CvqTn0"  # Replace with your actual API key
youtube = build("youtube", "v3", developerKey=API_KEY)

# Configure the Gemini API
gemini_key = "AIzaSyAJN1QH8PpI0V-_fUHZPuHtkYl5jHlNJT0"  # Replace with your actual Gemini API key
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('models/gemini-1.0-pro')

def youtube_search(query, max_results=5):
    """Searches YouTube for videos based on the query and returns only videos with transcripts."""
    results = []
    next_page_token = None

    while len(results) < max_results:
        search_results = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=10,
            pageToken=next_page_token,
            type="video",
            order="relevance"
        ).execute()

        for item in search_results['items']:
            video_id = item['id']['videoId']
            if has_transcript(video_id):
                results.append({
                    "title": item['snippet']['title'],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "video_id": video_id
                })
            if len(results) >= max_results:
                break

        next_page_token = search_results.get('nextPageToken')
        if not next_page_token:
            break

    return results

def has_transcript(video_id):
    """Returns True if a transcript is available for the given YouTube video ID."""
    try:
        YouTubeTranscriptApi.get_transcript(video_id)
        return True
    except Exception:
        return False

def get_transcript(video_id):
    """Fetches the transcript from a YouTube video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        st.error(f"Error getting transcript: {e}")
        return None

def compare_transcripts(transcripts, topic):
    """Compare transcripts using cosine similarity and return the best match."""
    best_score = 0
    best_transcript = None

    vectorizer = TfidfVectorizer().fit([topic] + transcripts)
    topic_vector = vectorizer.transform([topic])

    for transcript in transcripts:
        transcript_vector = vectorizer.transform([transcript])
        score = cosine_similarity(topic_vector, transcript_vector)[0][0]

        if score > best_score:
            best_score = score
            best_transcript = transcript

    return best_transcript

def generate_professional_script(transcript):
    """Generates a professionalized script based on the best transcript."""
    prompt = f"Convert this into an elaborated and understandable professionalized script: {transcript}"
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.title("YouTube Video Search Transcription")

# Search bar
query = st.text_input("Enter your search query:")

# Search results and process
if query:
    results = youtube_search(query, max_results=3)
    transcripts = []

    if results:
        for i, result in enumerate(results):
            st.subheader(f"Video {i+1}")
            st.write(f"Title: {result['title']}")
            st.write(f"URL: {result['url']}")

            # Extract video ID from the URL
            video_id = result['video_id']

            # Get transcript
            with st.spinner("Fetching transcript..."):
                transcript = get_transcript(video_id)
                if transcript:
                    st.write("Transcript:")
                    st.text_area(f"Transcript {i+1}", transcript, height=150)
                    transcripts.append(transcript)
                else:
                    st.error("Transcription failed.")
        
        if transcripts:
            # Compare transcripts based on the topic
            with st.spinner("Comparing transcripts..."):
                best_transcript = compare_transcripts(transcripts, query)
                if best_transcript:
                    st.write("Best Transcript based on the topic:")
                    st.text_area("Best Transcript:", best_transcript, height=150)
                else:
                    st.warning("Could not determine the best transcript.")
            
            # Add a button to generate the professional script
            if best_transcript:
                if st.button("Generate Script"):
                    with st.spinner("Generating script..."):
                        professional_script = generate_professional_script(best_transcript)
                        st.write("Script:")
                        st.text_area("Script:", professional_script, height=200)
    else:
        st.warning("No videos with transcripts found.")
