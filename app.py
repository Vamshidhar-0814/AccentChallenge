import streamlit as st
import os
import yt_dlp
import torch
from speechbrain.pretrained import EncoderClassifier
import ffmpeg
import logging

# Suppress some common warnings (optional, for cleaner console output)
logging.getLogger('ffmpeg').setLevel(logging.ERROR)
logging.getLogger('yt_dlp').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)

# Streamlit page configuration must be first
st.set_page_config(page_title="English Accent Classifier", page_icon="🎤", layout="centered")

st.title("English Accent Classifier")

st.markdown("""
This tool analyzes the English accent spoken in a public video URL (e.g., YouTube, Loom, direct MP4 links)
to help identify the geographical origin of the speaker's accent.
""")

# Helper function to extract audio using ffmpeg-python
def extract_audio_ffmpeg(video_path, audio_path="audio.wav"):
    try:
        ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k').run(overwrite_output=True)
    except ffmpeg.Error as e:
        st.error(f"FFmpeg error during audio extraction: {e.stderr.decode()}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        raise

# Removed @st.cache_data from here
def download_and_extract_audio(video_url, audio_path="audio.wav"):
    # Ensure audio.wav is cleaned up from previous runs if it exists
    # This is now crucial as the function is not cached and always re-creates
    if os.path.exists(audio_path):
        os.remove(audio_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded.%(ext)s', # This is just a template, actual filename can vary
        'quiet': True,
        'no_warnings': True,
    }
    video_filename = "" # Initialize to ensure it's defined
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            # Use ydl.prepare_filename to get the exact path of the downloaded video file
            video_filename = ydl.prepare_filename(info_dict)

        extract_audio_ffmpeg(video_filename, audio_path)

        # Check if audio file was actually created and is not empty
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Audio file was not created or is empty. Video might not contain an audio track or extraction failed silently.")

        return audio_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Failed to download video: {e}. Please check the URL or its accessibility.")
        return None
    except Exception as e:
        st.error(f"An error occurred during download or audio extraction: {e}")
        return None
    finally:
        # Clean up downloaded video file ONLY
        if os.path.exists(video_filename):
            os.remove(video_filename)


# 2️⃣ Load pre-trained accent classifier
@st.cache_resource(show_spinner="Loading AI model...")
def load_classifier():
    classifier = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa"
    )
    return classifier

classifier = load_classifier()

# Run classification
def classify_accent(audio_path):
    try:
        signal = classifier.load_audio(audio_path)
        prediction = classifier.classify_batch(signal)

        # prediction[1] typically holds the log-probabilities or raw scores
        raw_scores = prediction[1]
        if raw_scores.dim() == 1:
            raw_scores = raw_scores.unsqueeze(0)

        score = torch.softmax(raw_scores, dim=1)

        label = prediction[3][0]  # Predicted accent label (first element of the list of labels)
        confidence = score.max().item() * 100
        return label, confidence
    except Exception as e:
        st.error(f"Error during accent classification: {e}")
        return None, None


# Streamlit UI
with st.form("accent_form"):
    video_url = st.text_input(
        "Enter a public video URL:",
        placeholder="e.g., https://www.youtube.com/watch?v=your_video_id or a Loom link",
        key="video_input"
    )
    submit_button = st.form_submit_button("Analyze Accent")

# This block executes when the button is clicked
if submit_button:
    # Clear previous results if any
    # This helps in displaying clean results on re-run
    if 'last_label' in st.session_state:
        del st.session_state['last_label']
    if 'last_confidence' in st.session_state:
        del st.session_state['last_confidence']

    if not video_url:
        st.warning("Please enter a valid video URL.")
    else:
        audio_path = None # Initialize audio_path outside try-except
        try:
            with st.spinner("Downloading and extracting audio..."):
                audio_path = download_and_extract_audio(video_url)

            if audio_path:
                with st.spinner("Analyzing accent..."):
                    label, confidence = classify_accent(audio_path)

                if label and confidence:
                    st.success("Analysis complete!")
                    # Store results in session state to persist across reruns
                    st.session_state['last_label'] = label.upper()
                    st.session_state['last_confidence'] = confidence
                else:
                    st.error("Analysis could not be completed.")
            else:
                st.error("Download or audio extraction failed.")

        finally:
            # Ensure audio file is cleaned up after entire process,
            # even if an error occurs.
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

# Display results if they exist in session state
if 'last_label' in st.session_state and 'last_confidence' in st.session_state:
    st.markdown(f"## Detected Accent: **{st.session_state['last_label']}**")
    st.markdown(f"### Confidence: **{st.session_state['last_confidence']:.2f}%**")

    st.info("💡 **Why country names instead of 'American' or 'British' accent?**\n\n"
        "Accent classification models are often trained on datasets labeled with geographical regions (countries) "
        "where specific English accents are prevalent. While we colloquially refer to 'American' or 'British' accents, "
        "these are broad categories with many sub-variations within those countries. "
        "Using country names (e.g., 'US', 'UK', 'Australia') provides a more granular and direct representation "
        "of the training data's labels and helps in pinpointing a specific regional influence rather than a general term.")

st.caption("Built by Vamshidhar Reddy · [vamshidharreddy0814@gmail.com](mailto:vamshidharreddy0814@gmail.com)")