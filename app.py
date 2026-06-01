import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
import speech_recognition as sr
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

import streamlit.components.v1 as components



st.set_page_config(
    page_title="SwasthAI Ultra",
    layout="wide"
)


st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#000000,#020617,#001f3f);
    color:white;
}

.card{
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    border:1px solid #22c55e;
    box-shadow:0 0 20px rgba(34,197,94,0.3);
    margin-bottom:15px;
}

.stButton>button{
    background: linear-gradient(90deg,#22c55e,#00f0ff);
    color:black;
    border:none;
    border-radius:12px;
    font-weight:bold;
}

h1,h2,h3{
    color:#22c55e;
}

</style>
""", unsafe_allow_html=True)


def animated_ai_robot():

    components.html("""
    <style>

    body{
        margin:0;
        overflow:hidden;
    }

    .robot-container{
        display:flex;
        justify-content:center;
        align-items:center;
        height:520px;
        position:relative;
    }

    .robot{
        position:relative;
        width:260px;
        height:420px;
        animation:float 4s ease-in-out infinite;
    }

    .head{
        width:150px;
        height:130px;
        background:linear-gradient(145deg,#00e5ff,#0077ff);
        border-radius:30px;
        position:relative;
        margin:auto;
        box-shadow:
        0 0 20px #00f0ff,
        0 0 40px #00f0ff,
        0 0 80px #00f0ff;
        border:3px solid white;
    }

    .eye{
        width:28px;
        height:28px;
        background:#000;
        border-radius:50%;
        position:absolute;
        top:40px;
        overflow:hidden;
    }

    .eye::before{
        content:'';
        position:absolute;
        width:10px;
        height:10px;
        background:#00ffff;
        border-radius:50%;
        top:8px;
        left:8px;
        animation:scan 2s infinite;
    }

    .left{
        left:30px;
    }

    .right{
        right:30px;
    }

    .mouth{
        width:70px;
        height:12px;
        background:#111;
        position:absolute;
        bottom:25px;
        left:40px;
        border-radius:20px;
        overflow:hidden;
    }

    .mouth::before{
        content:'';
        position:absolute;
        width:100%;
        height:100%;
        background:#00ffcc;
        animation:talk 0.4s infinite;
    }

    .antenna{
        width:8px;
        height:40px;
        background:white;
        position:absolute;
        top:-40px;
        left:71px;
        border-radius:10px;
    }

    .antenna::before{
        content:'';
        position:absolute;
        width:18px;
        height:18px;
        background:red;
        border-radius:50%;
        top:-10px;
        left:-5px;
        box-shadow:0 0 20px red;
        animation:blink 1s infinite;
    }

    .body{
        width:180px;
        height:200px;
        margin:20px auto;
        border-radius:30px;
        background:linear-gradient(145deg,#021b2b,#033d5b);
        border:2px solid #00f0ff;
        box-shadow:
        0 0 20px #00f0ff,
        inset 0 0 20px #00f0ff;
        position:relative;
    }

    .core{
        width:90px;
        height:90px;
        border-radius:50%;
        background:radial-gradient(circle,#ff0000,#990000);
        position:absolute;
        top:50px;
        left:45px;
        box-shadow:
        0 0 20px red,
        0 0 50px red,
        0 0 90px red;
        animation:pulse 2s infinite;
    }

    .arm-left,
    .arm-right{
        width:35px;
        height:140px;
        background:#00e5ff;
        position:absolute;
        top:160px;
        border-radius:20px;
        box-shadow:0 0 20px #00f0ff;
    }

    .arm-left{
        left:-30px;
        animation:wave 2s infinite;
        transform-origin:top;
    }

    .arm-right{
        right:-30px;
    }

    .leg-left,
    .leg-right{
        width:40px;
        height:120px;
        background:#00e5ff;
        position:absolute;
        bottom:-100px;
        border-radius:20px;
        box-shadow:0 0 20px #00f0ff;
    }

    .leg-left{
        left:50px;
    }

    .leg-right{
        right:50px;
    }

    .ring{
        position:absolute;
        width:300px;
        height:300px;
        border:2px solid rgba(0,255,255,0.3);
        border-radius:50%;
        top:50px;
        left:-20px;
        animation:rotate 8s linear infinite;
    }

    .ring2{
        width:360px;
        height:360px;
        top:20px;
        left:-50px;
        animation-direction:reverse;
    }

    .ai-text{
        position:absolute;
        width:100%;
        bottom:-160px;
        text-align:center;
        color:#00ffff;
        font-size:22px;
        font-family:monospace;
        text-shadow:0 0 20px #00ffff;
        animation:flicker 2s infinite;
    }

    @keyframes float{
        0%{transform:translateY(0px);}
        50%{transform:translateY(-20px);}
        100%{transform:translateY(0px);}
    }

    @keyframes pulse{
        0%{transform:scale(1);}
        50%{transform:scale(1.15);}
        100%{transform:scale(1);}
    }

    @keyframes rotate{
        0%{transform:rotate(0deg);}
        100%{transform:rotate(360deg);}
    }

    @keyframes talk{
        0%{height:4px;}
        50%{height:12px;}
        100%{height:4px;}
    }

    @keyframes scan{
        0%{transform:translateX(-5px);}
        50%{transform:translateX(10px);}
        100%{transform:translateX(-5px);}
    }

    @keyframes blink{
        0%{opacity:1;}
        50%{opacity:0.2;}
        100%{opacity:1;}
    }

    @keyframes flicker{
        0%{opacity:1;}
        50%{opacity:0.6;}
        100%{opacity:1;}
    }

    @keyframes wave{
        0%{transform:rotate(0deg);}
        50%{transform:rotate(20deg);}
        100%{transform:rotate(0deg);}
    }

    </style>

    <div class="robot-container">

        <div class="robot">

            <div class="ring"></div>
            <div class="ring ring2"></div>

            <div class="head">

                <div class="antenna"></div>

                <div class="eye left"></div>
                <div class="eye right"></div>

                <div class="mouth"></div>

            </div>

            <div class="body">

                <div class="core"></div>

            </div>

            <div class="arm-left"></div>
            <div class="arm-right"></div>

            <div class="leg-left"></div>
            <div class="leg-right"></div>

            <div class="ai-text">
                SWASTHAI AI DOCTOR ACTIVE
            </div>

        </div>

    </div>

    <script>

    const messages = [
        "Hello Human. I am Agastya.",
        "Health Scan Activated.",
        "Please maintain a healthy lifestyle.",
        "Monitoring your health parameters.",
        "AI healthcare system is now online."
    ];

    let index = 0;

    function speakMessage(text){

        const speech = new SpeechSynthesisUtterance(text);

        speech.lang = "en-US";
        speech.volume = 1;
        speech.rate = 1;
        speech.pitch = 1.1;

        window.speechSynthesis.speak(speech);
    }

    window.onload = function(){

        speakMessage(messages[0]);

    };

    setInterval(()=>{

        const textBox = document.querySelector(".ai-text");

        textBox.innerHTML = messages[index];

        speakMessage(messages[index]);

        index = (index + 1) % messages.length;

    }, 7000);

    </script>

    """, height=650)

def speak(text):

    tts = gTTS(text=text, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:

        tts.save(fp.name)

        audio_file = open(fp.name, "rb")
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/mp3")



def load_data(path):

    df = pd.read_csv(path)

    df = pd.get_dummies(df, drop_first=True)

    df = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(df),
        columns=df.columns
    )

    return df

def train_model(df):

    target = df.columns[-1]

    X = df.drop(target, axis=1)
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, scaler, acc, X.columns


def explain_report(disease, risk, prob, features, model):

    importance = model.coef_[0]

    top = sorted(
        zip(features, importance),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    factors = ", ".join([f[0] for f in top])

    return f"""
🧾 AI Medical Report

Disease: {disease}

Risk Level: {risk}

Prediction Probability: {round(prob*100,2)}%

📊 Key Factors:
{factors}

💡 Interpretation:
You show {risk.lower()} indicators for {disease}

🩺 Advice:
✔ Maintain healthy diet
✔ Exercise regularly
✔ Monitor symptoms
✔ Consult doctor if symptoms persist

⚠ AI Generated Result
"""



def get_nutrition_plan(disease, diet_type, risk):

    factor = {
        "High":0.8,
        "Medium":1.0,
        "Low":1.2
    }[risk]

    veg = [
        ["6:30 AM","Warm water",1,0],
        ["8:00 AM","Oats + nuts",1,250],
        ["11:00 AM","Fruits",1,80],
        ["1:30 PM","Rice + dal",1,450],
        ["4:30 PM","Snacks",1,100],
        ["8:00 PM","Chapati + curry",1,300]
    ]

    nonveg = [
        ["6:30 AM","Warm water",1,0],
        ["8:00 AM","Eggs + oats",1,300],
        ["11:00 AM","Fruits",1,80],
        ["1:30 PM","Chicken + rice",1,500],
        ["4:30 PM","Snacks",1,120],
        ["8:00 PM","Fish + vegetables",1,350]
    ]

    plan = veg if diet_type == "veg" else nonveg

    final = []

    for t,m,q,c in plan:

        final.append([
            t,
            m,
            q,
            round(c*factor,1)
        ])

    return pd.DataFrame(
        final,
        columns=["Time","Meal","Quantity","Calories"]
    )



def get_ayurveda():

    herbs = [

        "Ashwagandha",
        "Tulsi",
        "Neem",
        "Turmeric",
        "Amla",
        "Fenugreek",
        "Giloy",
        "Arjuna",
        "Brahmi"

    ]

    return herbs




def get_ayurveda_diet_plan(disease, dosha):

    plans = {

        "Vata":[

            ["6:00 AM","Warm water + honey","1 glass",40],
            ["7:30 AM","Herbal tea + soaked almonds","5 almonds",120],
            ["9:00 AM","Oats porridge with ghee","1 bowl",250],
            ["11:30 AM","Sweet fruits","1 plate",100],
            ["1:30 PM","Rice + dal + vegetables","1 plate",450],
            ["4:30 PM","Tulsi tea","1 cup",40],
            ["7:30 PM","Khichdi + ghee","1 bowl",300],
            ["9:00 PM","Turmeric milk","1 glass",120]

        ],

        "Pitta":[

            ["6:00 AM","Coriander detox water","1 glass",20],
            ["7:30 AM","Coconut water","1 glass",60],
            ["9:00 AM","Fruit bowl + oats","1 bowl",220],
            ["11:30 AM","Cucumber salad","1 plate",80],
            ["1:30 PM","Chapati + dal + curry","2 chapati",420],
            ["4:30 PM","Mint tea","1 cup",35],
            ["7:30 PM","Vegetable soup + rice","1 bowl",280],
            ["9:00 PM","Amla juice","1 glass",70]

        ],

        "Kapha":[

            ["6:00 AM","Warm lemon water","1 glass",25],
            ["7:30 AM","Ginger tea","1 cup",35],
            ["9:00 AM","Millet upma","1 bowl",200],
            ["11:30 AM","Papaya","1 plate",90],
            ["1:30 PM","Brown rice + curry","1 plate",380],
            ["4:30 PM","Green tea","1 cup",20],
            ["7:30 PM","Vegetable soup","1 bowl",180],
            ["9:00 PM","Turmeric water","1 glass",30]

        ]
    }

    # DIABETES SPECIAL

    if disease == "Diabetes":

        plans["Kapha"].append(
            ["Before Breakfast","Fenugreek water","1 glass",15]
        )

        plans["Kapha"].append(
            ["After Lunch","Neem juice","50 ml",20]
        )

        plans["Kapha"].append(
            ["Evening","Giloy herbal drink","1 cup",25]
        )

    # HEART SPECIAL

    if disease == "Heart Disease":

        plans["Pitta"].append(
            ["Morning","Arjuna herbal drink","1 cup",35]
        )

        plans["Pitta"].append(
            ["Night","Garlic milk","1 glass",90]
        )

    return pd.DataFrame(
        plans[dosha],
        columns=["Time","Meal","Quantity","Calories"]
    )



def get_exercises(disease):

    return [
        "Walking",
        "Yoga",
        "Cycling",
        "Meditation",
        "Breathing Exercises"
    ]



def get_gym_plan():

    return pd.DataFrame([

        ["Monday","Walking","30 mins"],
        ["Tuesday","Cycling","20 mins"],
        ["Wednesday","Yoga","25 mins"],
        ["Thursday","Strength","20 mins"],
        ["Friday","Cardio","30 mins"]

    ], columns=["Day","Exercise","Duration"])




def chatbot(q):

    q = q.lower()

    emergency_keywords = [
        "chest pain",
        "heart attack",
        "stroke",
        "breathing difficulty",
        "unconscious"
    ]

    for word in emergency_keywords:

        if word in q:

            return """
🚨 EMERGENCY ALERT

Your symptoms may indicate a serious condition.

⚠ Possible Issue:
Heart attack / Stroke / Critical condition

🩺 Immediate Actions:
• Call emergency services immediately
• Stay calm and avoid movement
• Seek nearest hospital

❗ Do NOT ignore these symptoms
"""

    if "heart" in q or "bp" in q:

        return """
❤️ Heart Health Advice

🧾 Possible Causes:
High cholesterol, BP, stress

🥗 Diet:
• Low salt, low oil
• Fruits, oats, nuts

🏃 Exercise:
• Walking (30 mins)
• Light cardio

💊 Advice:
Monitor BP regularly
"""

    if "diabetes" in q or "sugar" in q:

        return """
🍬 Diabetes Management

🧾 Cause:
High blood sugar levels

🥗 Diet:
• Avoid sugar & refined carbs
• Eat fiber-rich foods

🏃 Exercise:
• Daily walking
• Weight control

💊 Advice:
Check glucose regularly
"""

    if "ayurveda" in q:

        return """
🌿 Ayurveda Advice

• Drink warm water
• Sleep early
• Use herbal foods
• Practice yoga daily
"""

    if "fever" in q:

        return """
🤒 Fever Advice

• Stay hydrated
• Take rest
• Use paracetamol if needed

⚠ If fever > 3 days consult doctor
"""

    if "tired" in q or "weak" in q:

        return """
⚡ Fatigue Advice

🧾 Possible Causes:
• Low nutrition
• Stress
• Anemia

🥗 Improve diet:
• Iron-rich foods
• Protein intake

🩺 Check blood levels if persistent
"""

    if "diet" in q:

        return """
🥗 General Diet Plan

• Breakfast: Oats + fruits
• Lunch: Rice + dal + vegetables
• Dinner: Light chapati + curry

✔ Drink water
✔ Avoid junk food
"""

    if "exercise" in q or "gym" in q:

        return """
🏃 Fitness Plan

• Walking – 30 mins
• Yoga – 20 mins
• Strength – 3x/week

✔ Stay consistent
"""

    return """
🧠 AI Doctor Response

I understand your query.

💡 General Advice:
• Maintain healthy diet
• Exercise regularly
• Monitor symptoms

👉 Please provide more details for accurate guidance
"""




page = st.sidebar.radio(
    "🧭 Navigation",
    [
        "Dashboard",
        "Prediction",
        "Chatbot",
        "Hospital",
        "Appointment"
    ]
)


datasets = {

    "Diabetes":"datasets/diabetes.csv",
    "Heart Disease":"datasets/heart.csv"

}


if page == "Dashboard":

    st.title("🧠 SwasthAI Ultra")

    st.subheader("🧬 AI Doctor Live System")

    animated_ai_robot()

    if "voice" not in st.session_state:

        speak("Hello I am your AI Doctor Mokshith. How can I help you today")

        st.session_state.voice = True

    st.markdown("""
    <div class="card">

    <h2>🚀 AI Healthcare Intelligence System</h2>

    ✔ AI Disease Prediction  
    ✔ Ayurveda Diet System  
    ✔ Nutrition Recommendation  
    ✔ Exercise Planning  
    ✔ AI Voice Assistant  
    ✔ Hospital Recommendation

    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Live Health Insights")

    chart = pd.DataFrame({

        "Metric":["Heart","Diabetes","Fitness"],
        "Value":[65,40,80]

    })

    st.bar_chart(chart.set_index("Metric"))

    st.subheader("⚡ System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Model", "Active", "↑ 100%")

    with col2:
        st.metric("Prediction Accuracy", "92%", "+2%")

    with col3:
        st.metric("Voice Engine", "Online", "Stable")

elif page == "Prediction":

    st.title("🩺 Disease Prediction")

    disease = st.selectbox(
        "Select Disease",
        list(datasets.keys())
    )

    diet_type = st.radio(
        "Diet Type",
        ["veg","non-veg"]
    )

    ayurveda_type = st.selectbox(
        "🌿 Ayurveda Body Type",
        ["Vata","Pitta","Kapha"]
    )

    df = load_data(datasets[disease])

    model, scaler, acc, features = train_model(df)

    st.success(f"Model Accuracy: {round(acc*100,2)}%")

    st.subheader("🧾 Enter Patient Data")

    inputs = []

    for f in features:

        val = st.number_input(
            f"Enter {f}",
            min_value=0.0,
            step=1.0
        )

        inputs.append(val)

    if st.button("Predict"):

        arr = scaler.transform(
            np.array(inputs).reshape(1,-1)
        )

        prob = model.predict_proba(arr)[0][1]

        pred = model.predict(arr)[0]

        if prob > 0.7:
            risk = "High"

        elif prob > 0.4:
            risk = "Medium"

        else:
            risk = "Low"

        st.subheader("⚠ Risk Analysis")

        st.error(f"Risk Level: {risk}")

        health_score = 100 - int(prob*100)

        st.subheader("💚 Health Score")

        st.progress(health_score)

        st.write(f"Health Score: {health_score}/100")

        st.subheader("🧾 AI Medical Report")

        report = explain_report(
            disease,
            risk,
            prob,
            features,
            model
        )

        st.markdown(
            f"<div class='card'>{report}</div>",
            unsafe_allow_html=True
        )

        # NORMAL DIET

        st.subheader("🥗 Nutrition Diet Plan")

        diet_df = get_nutrition_plan(
            disease,
            diet_type,
            risk
        )

        st.dataframe(diet_df)

        st.success(
            f"Total Calories: {diet_df['Calories'].sum()} kcal"
        )

        # AYURVEDA DIET

        st.subheader("🌿 Personalized Ayurveda Diet")

        ayurveda_df = get_ayurveda_diet_plan(
            disease,
            ayurveda_type
        )

        st.dataframe(ayurveda_df)

        st.success(
            f"Ayurveda Calories: {ayurveda_df['Calories'].sum()} kcal"
        )

        # AYURVEDA HERBS

        st.subheader("🌿 Ayurvedic Herbs")

        for herb in get_ayurveda():

            st.write("🌱", herb)

        # FOODS TO AVOID

        avoid = {

            "Vata":[
                "Cold drinks",
                "Dry foods",
                "Too much fasting"
            ],

            "Pitta":[
                "Spicy foods",
                "Fried foods",
                "Alcohol"
            ],

            "Kapha":[
                "Sugar",
                "Junk foods",
                "Heavy dairy"
            ]
        }

        st.subheader("🚫 Foods To Avoid")

        for item in avoid[ayurveda_type]:

            st.write("❌", item)

        # LIFESTYLE

        st.subheader("🧘 Ayurveda Lifestyle")

        lifestyle = {

            "Vata":[
                "Sleep early",
                "Meditation",
                "Avoid stress"
            ],

            "Pitta":[
                "Drink more water",
                "Avoid spicy food",
                "Stay calm"
            ],

            "Kapha":[
                "Daily exercise",
                "Avoid oversleeping",
                "Stay active"
            ]
        }

        for tip in lifestyle[ayurveda_type]:

            st.write("🌿", tip)

        # EXERCISES

        st.subheader("🏃 Exercises")

        for e in get_exercises(disease):

            st.write("✔", e)

        # GYM PLAN

        st.subheader("🏋 Gym Plan")

        st.dataframe(get_gym_plan())

        st.warning(
            "⚠ Consult a certified doctor before following any diet or medication."
        )



elif page == "Chatbot":

    st.title("💬 AI Medical Assistant")

    if "chat" not in st.session_state:

        st.session_state.chat = []

    msg = st.text_input(
        "Describe your symptoms or ask anything"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Send") and msg:

            res = chatbot(msg)

            st.session_state.chat.append(
                ("🧑 You", msg)
            )

            st.session_state.chat.append(
                ("🤖 Doctor", res)
            )

            speak(res)

    with col2:

        if st.button("🎤 Say Hey Doctor"):

            r = sr.Recognizer()

            try:

                with sr.Microphone() as src:

                    st.info("Speak now...")

                    audio = r.listen(src)

                text = r.recognize_google(audio)

                st.write("You:", text)

                res = chatbot(text)

                st.session_state.chat.append(
                    ("🧑 You", text)
                )

                st.session_state.chat.append(
                    ("🤖 Doctor", res)
                )

                speak(res)

            except:

                st.error("Voice error")

    for sender, message in st.session_state.chat:

        st.markdown(
            f"""
            <div class='card'>
            <b>{sender}</b><br><br>
            {message}
            </div>
            """,
            unsafe_allow_html=True
        )


elif page == "Hospital":

    st.title("🏥 Top Hospitals in India")

    city = st.text_input(
        "Enter City (e.g. Hyderabad, Chennai, Delhi, Mumbai, Bangalore)"
    )

    hospitals = {

        "hyderabad":[
            {
                "name":"Apollo Hospitals",
                "speciality":"Heart, Diabetes",
                "address":"Jubilee Hills",
                "contact":"+91-40-23607777",
                "website":"https://www.apollohospitals.com"
            },
            {
                "name":"Yashoda Hospitals",
                "speciality":"Cardiology, Diabetes",
                "address":"Somajiguda",
                "contact":"+91-40-45674567",
                "website":"https://www.yashodahospitals.com"
            }
        ],

        "bangalore":[
            {
                "name":"Manipal Hospital",
                "speciality":"Heart, Diabetes",
                "address":"Old Airport Road",
                "contact":"+91-80-25024444",
                "website":"https://www.manipalhospitals.com"
            },
            {
                "name":"Fortis Hospital",
                "speciality":"Cardiac Care",
                "address":"Bannerghatta Road",
                "contact":"+91-80-66214444",
                "website":"https://www.fortishealthcare.com"
            }
        ],

        "chennai":[
            {
                "name":"Apollo Hospitals Greams Road",
                "speciality":"Heart Institute, Diabetes",
                "address":"Greams Lane",
                "contact":"+91-44-28293333",
                "website":"https://www.apollohospitals.com"
            }
        ],

        "delhi":[
            {
                "name":"AIIMS Delhi",
                "speciality":"Cardiology, Diabetes",
                "address":"Ansari Nagar",
                "contact":"+91-11-26588500",
                "website":"https://www.aiims.edu"
            }
        ],

        "mumbai":[
            {
                "name":"Kokilaben Hospital",
                "speciality":"Heart, Diabetes",
                "address":"Andheri",
                "contact":"+91-22-30999999",
                "website":"https://www.kokilabenhospital.com"
            }
        ]
    }

    if st.button("Search"):

        if city.lower() in hospitals:

            for h in hospitals[city.lower()]:

                st.markdown(f"""
                <div class='card'>

                <h3>🏥 {h['name']}</h3>

                📍 <b>Address:</b> {h['address']} <br><br>

                ❤️ <b>Speciality:</b> {h['speciality']} <br><br>

                📞 <b>Contact:</b> {h['contact']} <br><br>

                🔗 <a href="{h['website']}" target="_blank">
                Visit Website
                </a>

                </div>
                """, unsafe_allow_html=True)

        else:

            st.warning("No hospitals found for this city")
# =========================================================
# APPOINTMENT BOOKING PAGE
# =========================================================

elif page == "Appointment":

    st.title("🏥 Book Doctor Appointment")

    st.markdown("""
    <div class='card'>

    <h2>🩺 Smart AI Appointment System</h2>

    ✔ Book appointment by disease  
    ✔ AI specialist recommendation  
    ✔ Hospital selection  
    ✔ Time slot booking  

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # PATIENT DETAILS
    # =====================================================

    st.subheader("👤 Patient Information")

    patient_name = st.text_input(
        "Enter Patient Name"
    )

    phone = st.text_input(
        "Enter Phone Number"
    )

    age = st.number_input(
        "Enter Age",
        min_value=1,
        max_value=120,
        step=1
    )

    gender = st.selectbox(
        "Select Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    # =====================================================
    # DISEASE SELECTION
    # =====================================================

    disease_type = st.selectbox(
        "🩺 Select Disease",
        [
            "Heart Disease",
            "Diabetes",
            "Cancer",
            "Stroke",
            "Kidney Disease",
            "Lung Disease",
            "Fever",
            "Skin Disease"
        ]
    )

    # =====================================================
    # AI SPECIALIST RECOMMENDATION
    # =====================================================

    specialist_map = {

        "Heart Disease":"Cardiologist ❤️",

        "Diabetes":"Diabetologist 🍬",

        "Cancer":"Oncologist 🎗",

        "Stroke":"Neurologist 🧠",

        "Kidney Disease":"Nephrologist 🩺",

        "Lung Disease":"Pulmonologist 🫁",

        "Fever":"General Physician 🌡",

        "Skin Disease":"Dermatologist ✨"
    }

    specialist = specialist_map[disease_type]

    st.success(
        f"👨‍⚕ Recommended Specialist: {specialist}"
    )

    # =====================================================
    # HOSPITAL LIST
    # =====================================================

    hospital_data = {

        "Heart Disease":[
            "Apollo Hospitals",
            "Fortis Hospital",
            "Narayana Health"
        ],

        "Diabetes":[
            "Apollo Hospitals",
            "Manipal Hospital",
            "Aster CMI"
        ],

        "Cancer":[
            "Tata Memorial Hospital",
            "Kidwai Cancer Institute",
            "AIIMS Delhi"
        ],

        "Stroke":[
            "NIMHANS",
            "Apollo Hospitals",
            "Fortis Hospital"
        ],

        "Kidney Disease":[
            "Manipal Hospital",
            "Aster CMI",
            "Apollo Hospitals"
        ],

        "Lung Disease":[
            "Fortis Hospital",
            "Narayana Health",
            "AIIMS Delhi"
        ],

        "Fever":[
            "Apollo Clinic",
            "Aster Clinic",
            "Fortis Clinic"
        ],

        "Skin Disease":[
            "Kaya Skin Clinic",
            "Apollo Clinic",
            "Manipal Hospital"
        ]
    }

    hospital = st.selectbox(
        "🏥 Select Hospital",
        hospital_data[disease_type]
    )

    # =====================================================
    # DATE & TIME
    # =====================================================

    appointment_date = st.date_input(
        "📅 Select Appointment Date"
    )

    appointment_time = st.selectbox(
        "⏰ Select Time Slot",
        [
            "09:00 AM",
            "10:00 AM",
            "11:00 AM",
            "12:00 PM",
            "02:00 PM",
            "03:00 PM",
            "04:00 PM",
            "05:00 PM"
        ]
    )

    # =====================================================
    # BOOK BUTTON
    # =====================================================

    if st.button("✅ Confirm Appointment"):

        if patient_name == "" or phone == "":

            st.error(
                "⚠ Please fill all details"
            )

        else:

            st.balloons()

            st.success(
                "🎉 Appointment Booked Successfully"
            )

            st.markdown(f"""
            <div class='card'>

            <h2>🧾 Appointment Receipt</h2>

            👤 <b>Patient:</b> {patient_name}<br><br>

            🎂 <b>Age:</b> {age}<br><br>

            ⚧ <b>Gender:</b> {gender}<br><br>

            🩺 <b>Disease:</b> {disease_type}<br><br>

            👨‍⚕ <b>Specialist:</b> {specialist}<br><br>

            🏥 <b>Hospital:</b> {hospital}<br><br>

            📅 <b>Date:</b> {appointment_date}<br><br>

            ⏰ <b>Time:</b> {appointment_time}<br><br>

            📞 <b>Phone:</b> {phone}<br><br>

            ✔ Please arrive 15 minutes early.

            </div>
            """, unsafe_allow_html=True)

            speak(
                f"Appointment booked successfully for {patient_name}"
            )


st.write("🚀 SwasthAI Ultra - AI Healthcare System")