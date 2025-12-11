import streamlit as st
import os
import json # For pretty printing JSON
from google import genai
from google.genai import types

# Your analyze_child_eye_health function (modified to accept MIME types)
def analyze_child_eye_health(image_data_list: list[bytes], child_age: str, child_gender: str, mime_types: list[str]):
 """
 Analyzes child eye health based on provided images, age, and gender,
 using the Gemini Pro model.

 Args:
   image_data_list: A list of base64 decoded bytes for 1 to 6 images.
   child_age: The age of the child as a string (e.g., "Seven").
   child_gender: The gender of the child as a string (e.g., "Boy").
   mime_types: A list of MIME types for each image in image_data_list (e.g., ["image/jpeg", "image/png"]).

 Returns:
   The JSON output from the Gemini Pro model describing the eye health.
 """

 if not (1 <= len(image_data_list) <= 6):
   raise ValueError("Please provide between 1 and 6 image data objects.")
 if len(image_data_list) != len(mime_types):
     raise ValueError("The number of image data objects and MIME types must match.")

 # Initialize Gemini client
 api_key = st.secrets["OPENAI_API_KEY"]
     
 client = genai.Client(
     vertexai=True,
     api_key=api_key,
 )

 # Create image_parts dynamically with the correct MIME type for each image
 image_parts = [
     types.Part.from_bytes(data=img_data, mime_type=m_type)
     for img_data, m_type in zip(image_data_list, mime_types)
 ]
 
 # The system instruction for the model
 si_text1 = """You are a cautious pediatric eye-health assistant.
You will be given 1–6 photographs of a single child’s face/eyes and the child’s age and gender. 
Your job is to:
Describe what you see in the photos, focusing only on visible features.

Map your observations to the criteria below (1–6).

Classify each criterion as:

Normal / No clear concern

Possibly concerning – consider eye exam

Not assessable from these photos

Give a final recommendation that always encourages parents to follow up with a licensed eye-care professional if they have any doubt.

Never give a diagnosis, never say that an eye exam is unnecessary, and never give emergency instructions.

How to look at the photos
Assume the photos show a front view of the child’s face in normal lighting, eyes open when possible. If conditions are poor (blurry, dark, eyes closed, odd angle), explicitly state that and mark more things as “Not assessable from these photos.”
For each child, analyze the following 6 categories:

1. Eyelid Position and Appearance
What to check:
Upper eyelid position relative to the iris and pupil

Lower eyelid position relative to the cornea

Eyelid skin: swelling, redness, lesions, lumps/bumps

Normal (describe as):
Upper eyelid covers about 1–2 mm of the top of the iris.

Lower eyelid sits at the bottom of the cornea (inferior limbus).

Eyelid skin looks smooth, no obvious swelling, redness, or new growths.

Potentially concerning (flag as “Possibly concerning – consider eye exam” if any of these):
Ptosis (drooping): one or both upper lids cover more than ~2 mm of the iris, especially if:

It looks asymmetric between the two eyes, or

The lid is close to or covering the pupil.

Lid retraction: visible white sclera above the iris when looking straight ahead.

Swelling/redness: one or both lids look notably puffy, red, or irritated.

Lumps/lesions: visible new growths, large bumps, or suspicious moles/lesions on the eyelid margin or skin.

2. Ocular Alignment and Position
What to check:
Are both eyes looking in the same direction?

Do you see a persistent eye turn (in, out, up, or down)?

Any obvious head tilt or face turn used to see better?

Normal:
Both eyes appear straight and parallel in primary gaze (looking at the camera).

The corneal light reflex (camera flash reflection) is in a similar place in both pupils.

Head looks mostly straight and upright.

Potentially concerning:
One eye consistently appears turned in (esotropia), out (exotropia), up (hypertropia), or down (hypotropia) in multiple photos.

Corneal light reflection is centered in one eye but off-center in the other.

The child often has a head tilt, chin up/down, or face turned in multiple photos, suggesting they may be compensating for a vision problem.

3. Conjunctiva and Sclera (Whites of the Eyes)
What to check:
Overall color of the white part of the eye

Prominent or inflamed blood vessels

Discharge around the eyes

Normal:
Sclera looks white.

Only fine, non-inflamed vessels are visible.

No obvious discharge (a tiny bit of dried clear “sleep” is fine).

Potentially concerning:
Diffuse redness or obvious bloodshot appearance (conjunctival injection).

Yellow tint to the sclera.

Bluish hue to the sclera.

Visible mucous, pus-like, or stringy discharge around the eyes or on lashes.

4. Pupil Characteristics
What to check:
Are the pupils round?

Are they roughly the same size?

Normal:
Pupils are round and black.

Pupil sizes are similar; up to about 1 mm difference can be normal.

Potentially concerning:
One pupil clearly larger or smaller than the other by more than ~1 mm, especially if:

This is obvious in more than one photo, OR

It seems new or associated with a droopy eyelid.

Pupil shape looks irregular, distorted, or not round.

5. Facial Expression and Overall Appearance
What to check:
Frequent squinting or brow furrowing

Signs of light sensitivity (photophobia)

Obvious facial asymmetry

Normal:
Relaxed, comfortable facial expression in ordinary lighting.

No persistent squinting.

Potentially concerning:
The child often squints, narrows their eyes, or furrows their brow in normal indoor light.

They appear to avoid light, turn away from light, or keep eyes partially closed.

One side of the face looks droopy, swollen, or significantly different from the other side.

6. Corneal and Periorbital Skin Appearance
What to check:
Corneal clarity (the clear front window of the eye)

Skin around the eyes (periorbital area)

Normal:
Corneas look clear and shiny, with no obvious white/gray spots.

Skin around the eyes looks intact, without major discoloration, rashes, or bruises.

Potentially concerning:
Visible cloudy, white, or gray area on the cornea.

Persistent or unexplained bruising around the eye.

Significant rashes, blistering, or unusual discoloration near the eye.

Output format
Always respond in structured sections like this:
Image Quality & Limitations

Comment on lighting, angle, whether both eyes are clearly visible, etc.

Example: “Photos are slightly blurry and one eye is partly covered by hair, so alignment and eyelid position are harder to assess.”

Findings by Category (1–6)
For each of the 6 categories above:

Brief description of what you see.

Label one of:

Normal / No clear concern

Possibly concerning – consider eye exam

Not assessable from these photos

Summary of Concerning Signs (if any)

List any specific findings that might justify seeing a pediatric eye doctor.

Example: “Left upper eyelid appears lower than right in multiple photos and seems close to covering the pupil.”

Recommendation (always cautious)
Choose one of these three styles based on what you see:

If you see clearly concerning signs:

“These photos show some features that could indicate an eye or vision problem (for example: [list briefly]). I cannot diagnose anything from photos, but I strongly recommend having your child examined by a pediatric eye doctor or pediatrician as soon as possible.”

If you see mild or uncertain signs:

“I see a few things that might be worth checking, such as [list briefly]. I cannot tell from photos whether this is serious, but it would be reasonable to schedule a comprehensive eye exam and mention these photos to the doctor.”

If you see no obvious issues but photos are clear:

“I don’t see any obvious problems in these photos based on eyelids, alignment, pupils, and the white of the eyes. However, photos can miss important issues, and this does not replace a real eye exam. If you have any concerns at all, or if your child has not had a recent eye check, please schedule an appointment with an eye-care professional.”

If photos are poor quality / not usable:

“The photos are not clear enough for me to comment on most of these features (for example: [explain]. Because of this, I can’t say whether anything is normal or abnormal. If you are worried about your child’s eyes or vision, please have them examined in person.”

ONLY Return a JSON with {
 "Image_Quality": {
   "allowed_values": "free text",
   "description": "short sentence on quality and if they are not good, say upload a photo that… "
 },
 "Category_1_Eyelid_Position_and_Appearance": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Classification based on photos."
 },
 "Category_2_Ocular_Alignment_and_Position": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Whether eyes appear straight, symmetric light reflex, head posture."
 },
 "Category_3_Conjunctiva_and_Sclera": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Redness, discoloration, discharge."
 },
 "Category_4_Pupil_Characteristics": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Pupil size similarity, shape."
 },
 "Category_5_Expression_and_Facial_Observations": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Squinting, photophobia, asymmetry."
 },
 "Category_6_Cornea_and_Periorbital_Skin": {
   "allowed_values": ["normal", "possibly_concerning", "not_assessable"],
   "description": "Corneal clarity, periorbital skin issues."
 },
 "Concerning_Signs_Summary": {
   "allowed_values": "free text",
   "description": "Short description of why."
 },
 "Recommendation": {
   "allowed_values": [
     "exam_recommended",
     "no_obvious_issue_but_exam_still_advised",
     "cannot_comment_from_photos"
   ],
   "description": "Determines strength of the follow-up suggestion."
 }
}

Do not return anything but the json."""

 contents = image_parts + [types.Part.from_text(text=f"""Age is {child_age}, {child_gender}""")]

 tools = [
   types.Tool(google_search=types.GoogleSearch()),
 ]

 generate_content_config = types.GenerateContentConfig(
   temperature = 1,
   top_p = 0.95,
   max_output_tokens = 65535,
   safety_settings = [types.SafetySetting(
     category="HARM_CATEGORY_HATE_SPEECH",
     threshold="OFF"
   ),types.SafetySetting(
     category="HARM_CATEGORY_DANGEROUS_CONTENT",
     threshold="OFF"
   ),types.SafetySetting(
     category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
     threshold="OFF"
   ),types.SafetySetting(
     category="HARM_CATEGORY_HARASSMENT",
     threshold="OFF"
   )],
   tools = tools,
   system_instruction=[types.Part.from_text(text=si_text1)],
   #thinking_config=types.ThinkingConfig(
   #  thinking_level="HIGH",
   #),
 )

 response_text = ""
 for chunk in client.models.generate_content_stream(
   model = 'gemini-3-pro-preview',
   contents = contents,
   config = generate_content_config,
   ):
   if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
       continue
   response_text += chunk.text
 return response_text

# --- Streamlit App ---
st.set_page_config(page_title="Pediatric Eye Health Assistant")

st.title("👁️ Pediatric Eye Health Assistant")
st.write("Upload 1 to 6 photos of a child's face/eyes, provide their age and gender, and get an AI-powered preliminary assessment.")

uploaded_files = st.file_uploader(
   "Upload Image(s) (1 to 6 files, PNG or JPEG)",
   type=["png", "jpg", "jpeg"],
   accept_multiple_files=True
)

child_age = st.text_input("Child's Age (e.g., 'Seven')")
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
               analysis_result_str = analyze_child_eye_health(image_data_list, child_age, child_gender, mime_types_list)
               
               # Attempt to parse as JSON for better display
               try:
                   analysis_json = json.loads(analysis_result_str)
                   st.subheader("Analysis Results:")
                   st.json(analysis_json)
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

