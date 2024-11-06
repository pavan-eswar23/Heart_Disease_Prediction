import streamlit as st
import pickle
from apikey import google_gemini_api_keys
import os
import google.generativeai as genai

# Building an Generative AI - Model which helps patients in giving suggestions
genai.configure(api_key=google_gemini_api_keys)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-002",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)
chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Generate a Comprehensive engaging blog post relevant to the given title \"Effects of Generative Ai\" and keywords \"Artificial Creativity , Ethical Implication , Technology Innovation , Machine Learning Applications , AI Impact on Society\". Make sure to incorporate these keywords in the blog post . The Blog should be Approximately( num_words ) words in length ,Suitable for an online audience(Ensure the Content is Original ,informative and maintains a consistent tone throughout.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "**Effects of Generative AI: Shaping Society and Technology**\n\n**Introduction**\n\nThe advent of Generative Artificial Intelligence (AI) has sparked a profound transformation across diverse sectors. This technology, empowered by machine learning algorithms, enables the creation of novel content, ranging from art and music to text and code. As Generative AI continues to evolve, it is imperative to explore its multifaceted effects on creativity, ethics, innovation, and society as a whole.\n\n**Artificial Creativity and the Impact on Human Expression**\n\nGenerative AI has emerged as a powerful tool for expanding the boundaries of human creativity. AI-generated content can complement and enhance human expression, fostering new ideas and sparking inspiration. However, it also raises questions about the role of originality and the value of human creativity. As AI-generated works become increasingly sophisticated, it is crucial to establish clear guidelines and foster conversations on authenticity and attribution.\n\n**Ethical Implications and the Responsibility of AI**\n\nThe rapid development of Generative AI has brought to the forefront ethical concerns. The ability of AI to generate deepfakes and spread misinformation poses a significant threat to trust and transparency. It is essential to develop robust ethical frameworks and regulations to ensure responsible use and prevent malicious applications of Generative AI.\n\n**Technology Innovation and the Role of Machine Learning**\n\nGenerative AI is a testament to the transformative power of machine learning. Machine learning algorithms learn patterns and generate new content based on vast datasets. This technology has the potential to revolutionize industries, from healthcare to entertainment, by automating tasks, improving efficiency, and creating novel solutions.\n\n**Applications in the Real World and the Impact on Society**\n\nThe applications of Generative AI are far-reaching and touch upon various aspects of our lives. In medicine, AI-generated models can aid in disease diagnosis and drug discovery. In education, personalized content and tailored learning experiences can be created. The impact of Generative AI on society is profound, with potential benefits ranging from enhanced productivity to improved access to information.\n\n**Conclusion**\n\nGenerative AI is a transformative force that has profound effects on creativity, ethics, innovation, and society. While it offers immense potential for advancing technology and human expression, it also presents challenges that must be carefully addressed. By fostering meaningful conversations, developing ethical guidelines, and leveraging the technology responsibly, we can harness the power of Generative AI to create a future that is both innovative and equitable.",
      ],
    },
  ]
)
st.set_page_config(layout="wide")


#loading the deployed Model
heart_model= pickle.load(open('heart_disease_model.pkl','rb'))

st.title('Heart Disease Prediction🫀🩺')

# Create input fields for user data
st.header('Enter Patient Details:')
age = st.number_input('Age', min_value=0, max_value=150, value=40)
sex = st.selectbox('Sex', ['Male', 'Female'])  # Assuming sex is categorical
cp = st.number_input('Chest Pain Type', min_value=0, max_value=4, value=0)
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])  # Assuming fbs is categorical
restecg = st.number_input('Resting Electrocardiographic Results', min_value=0, max_value=3, value=0)
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])  # Assuming exang is categorical
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=0.0)
slope = st.number_input('Slope of the Peak Exercise ST Segment', min_value=0, max_value=3, value=0)
ca = st.number_input('Number of Major Vessels (0-3) Colored by Flourosopy', min_value=0, max_value=3, value=0)
thal = st.number_input('Thalassemia', min_value=0, max_value=3, value=0)

# Convert categorical inputs to numerical (if needed)
if sex == 'Male':
    sex = 1
else:
    sex = 0
if fbs == 'True':
    fbs = 1
else:
    fbs = 0
if exang == 'Yes':
    exang = 1
else:
    exang = 0

new_data = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]] 


if st.button('Predict'):
    # Make a prediction using your model
    prediction = heart_model.predict(new_data)
    print(prediction)

    # Display the prediction
    if prediction[0] == 0:
        st.success('The model predicts a low risk of heart disease.')
    else:
        st.error('The model predicts a high risk of heart disease.')
        prompt_parts=['Give some healthy tips regarding diet and fitness to the people having risk of heart disease']
        st.write("Don't Worry if you have chance of getting heart disease. Please follow the below tips to reduce the risk of getting heart disease.")
        response = chat_session.send_message(prompt_parts)
        st.write(response.text)
        #st.write(tips_message)


# Add some information or instructions 
#st.markdown('This is a simple demo for heart disease prediction. Please consult a healthcare professional for proper diagnosis and treatment.')


# Make a prediction using the loaded model