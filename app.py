import streamlit as st
import pdfplumber
from transformers import pipeline
import requests
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Your Google Gemini API key
API_KEY = "AIzaSyCDl-cPlgtAkLCRwA0JQ3C77Wa5DOeOgLE"

# Initialize the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        return "No text found in the PDF."

    # Split the text into manageable chunks if it's too long
    max_chunk_size = 1000  # Adjust this based on the model's token limit
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summaries = []
    for chunk in chunks:
        try:
            input_length = len(chunk.split())  # Count words to determine input length
            # Set max_length dynamically based on input length
            dynamic_max_length = min(180, max(30, input_length // 2))  # Adjust as needed
            summary = summarizer(chunk, max_length=dynamic_max_length, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error during summarization: {str(e)}")

    final_summary = " ".join(summaries)
    return final_summary

def query_chatbot(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }

    # Construct the data payload according to Gemini API specification
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        json_response = response.json()
        
        # Extract and return only the relevant content
        chatbot_content = json_response['candidates'][0]['content']['parts'][0]['text']
        return chatbot_content
    except requests.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e.response.text)
        return f"Error: {e.response.status_code}, {e.response.text}"
    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}"


st.title("PDF Summarizer and Chatbot")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    summary = summarize_pdf(uploaded_file)
    st.subheader("Summary:")
    st.write(summary)

st.subheader("Chat with the Bot")
user_query = st.text_input("Ask me anything about the PDF or related topics:")

if user_query:
    chatbot_response = query_chatbot(user_query)
    st.write(chatbot_response)
