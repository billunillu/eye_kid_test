import streamlit as st
import os
import json # For pretty printing JSON
from google import genai
from google.genai import types

# Your analyze_child_eye_health function (modified to accept MIME types)
def analyze_child_eye_health(image_data_list: list[bytes], child_age: str, child_gender: str, mime_types: list[str]):

 #api_key = st.secrets["API_KEY"]
 api_key = st.secrets["API_KEY"]
 store_name = 'fileSearchStores/eyeknowledge-tfe13blubv8d'
   
 client = genai.Client(
     api_key=api_key,
 )
 with open("prompt.txt", "rb") as f:
     prompt = f.read()
 # Create image_parts dynamically with the correct MIME type for each image
 image_parts = [
     types.Part.from_bytes(data=img_data, mime_type=m_type)
     for img_data, m_type in zip(image_data_list, mime_types)
 ]
 

 contents = image_parts + [types.Part.from_text(text=f"""Age is {child_age}, {child_gender}""")]

 generate_content_config=types.GenerateContentConfig(
    tools=[
        types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[store_name],
            )
        )
    ], system_instruction=[types.Part.from_text(text=prompt)]) # <--- This is now correctly placed))
 
 response_text = ""
 for chunk in client.models.generate_content_stream(
   model = 'gemini-3-pro-preview',
   contents = contents,
   config = generate_content_config,
   ):
   if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
       continue
   response_text += chunk.text
   token = response.usage_metadata.total_token_count
 return response_text,token

# --- Streamlit App ---
st.set_page_config(page_title="Pediatric Eye Health Assistant")

st.title("👁️ Pediatric Eye Health Assistant")
st.write("Upload 1 to 6 photos of a child's face/eyes, provide their age and gender, and get an AI-powered preliminary assessment.")

uploaded_files = st.file_uploader(
   "Upload Image(s) (1 to 6 files, PNG or JPEG)",
   type=["png", "jpg", "jpeg"],
   accept_multiple_files=True
)

child_age = st.text_input("Child's Age")
child_gender = st.selectbox("Child's Gender", ["Male", "Female", "Other", "Prefer not to say"])

if st.button("Analyze Eye Health"):
   if not uploaded_files:
       st.warning("Please upload at least one image.")
   elif not child_age:
       st.warning("Please enter the child's age.")
   else:
       with st.spinner("Analyzing photos... This may take a moment."):
           image_data_list = []
           mime_types_list = []
           for uploaded_file in uploaded_files:
               image_data_list.append(uploaded_file.read())
               mime_types_list.append(uploaded_file.type)

           try:
               # Call your analysis function
               analysis_result = analyze_child_eye_health(image_data_list, child_age, child_gender, mime_types_list)
               analysis_result_str = analysis_result[0]
               analysis_result_token = analysis_result[1]
               # Attempt to parse as JSON for better display
               try:
                   analysis_json = json.loads(analysis_result_str)
                   st.subheader("Analysis Results:")
                   st.json(analysis_json)
                   st.subheader("token use")
                   st.write(analysis_result_token)

                   st.subheader("cost estimate")
                   st.write(.15/1000000*analysis_result_token)
                   
               except json.JSONDecodeError:
                   st.subheader("Raw Analysis Output (not valid JSON):")
                   st.write(analysis_result_str)

           except Exception as e:
               st.error(f"An error occurred during analysis: {e}")
               st.error("Please ensure your GOOGLE_CLOUD_API_KEY is correctly set and the images are valid.")

st.markdown("---")
st.markdown(
   """
   **Disclaimer:** This assistant provides preliminary observations based on uploaded photos and
   should *never* be used as a substitute for professional medical advice or diagnosis.
   Always consult with a licensed eye-care professional for any health concerns.
   """
)

