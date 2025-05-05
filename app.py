import streamlit as st
import torch
import os
import sys
import asyncio

# Set page title and configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Fix for asyncio event loop errors with Streamlit on Windows
if os.name == 'nt':  # Windows
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load model using Streamlit's caching
@st.cache_resource
def load_model():
    try:
        st.info("Loading BERT sentiment model... This may take a moment.")
        from transformers import pipeline
        
        # Create the pipeline with the model
        classifier = pipeline('text-classification', model='bert-base-uncased-sentiment-model')
        st.success("Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main app title
st.title("Twitter Sentiment Analysis")
st.write("Analyze the sentiment of Twitter tweets using a fine-tuned BERT model.")

# Load the sentiment classifier
classifier = load_model()

if classifier is None:
    st.error("Could not load the model. Please check the logs for more information.")
else:
    # Main content area for text input
    text = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type your tweet here..."
    )
    
    # Process the text when the button is clicked
    if st.button("Analyze Sentiment", key="analyze_button_main") and text:
        with st.spinner("Analyzing sentiment..."):
            try:
                # Get prediction from model
                result = classifier(text)
                
                # Display raw model output
                st.subheader("Raw Model Output")
                st.json(result)
                
                # Extract key information for display
                if result and len(result) > 0:
                    label = result[0].get("label", "")
                    score = result[0].get("score", 0.0)
                    
                    # Display in a more readable format
                    st.subheader("Simplified Result")
                    st.markdown(f"**Predicted Emotion:** {label}")
                    st.markdown(f"**Confidence Score:** {score:.4f}")
                    
                    # Visualize the confidence
                    st.progress(float(score))
                
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
    
    elif not text and st.button("Analyze Sentiment", key="analyze_button_empty"):
        st.warning("Please enter some text to analyze.")
    
