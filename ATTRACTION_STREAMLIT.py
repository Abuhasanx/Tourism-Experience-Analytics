import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

#SET PAGE NAME
st.set_page_config(page_title="Tourism Attraction Recommender", layout="centered")

#STYLING THE APPLICATION WITH CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-container {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
        }
        .stButton>button {
            background-color: #4B9EFF;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #e43f3f;
        }
        .recommendation-box {
                background: #f0f8ff;
                padding: 12px;
                margin-top: 15px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 500;
                border-left: 5px solid #1e3a8a;
                color: black !important; /* Ensure text is visible */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar: App Information
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
🌍 **Discover Your Next Destination!**  
Get personalized **tourist spot** recommendations with ease.  

🔹 **How it Works?**  
1️⃣ Choose your **User ID**  
2️⃣ Click **"🔍 Find Attractions"**  
3️⃣ Explore **Top 5 Picks** just for you!  

💡 Powered by **AI & Machine Learning**  
""")


#LOAD DATA AND TRAINED MODEL
svd = joblib.load(r"D:\1final ds\svd.plk")
knn_model = joblib.load(r"D:\1final ds\knn.plk")
user_attraction_matrix = joblib.load(r"D:\1final ds\useratt2.plk")
user_attraction_matrix_reduced = joblib.load(r"D:\1final ds\user_att.plk")

st.markdown(
    """
    <style>
        /* Background and Main Container */
        body {
            background: linear-gradient(to right, #dfe9f3, #ffffff);
            font-family: 'Arial', sans-serif;
        }
        .main-container {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #4a90e2, #1e3a8a);
            color: white;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4a90e2, #1e3a8a);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 18px;
            border: none;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1e3a8a, #4a90e2);
            transform: scale(1.05);
        }

        /* Recommendation Box */
        .recommendation-box {
            background: #f0f8ff;
            padding: 12px;
            margin-top: 15px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
            border-left: 5px solid #1e3a8a;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# CREATE SELECT BOX FOR INPUT
st.markdown("<h3>Select User ID for Recommendations</h3>", unsafe_allow_html=True)
user_ids = user_attraction_matrix.index.tolist()
user_id = st.selectbox("Choose a User ID", user_ids)

#RECOMMENDATION FUNCTION 
def recommend_attractions(user_id, num_recommendations=5):
    user_idx = user_attraction_matrix.index.get_loc(user_id)
    
    distances, indices = knn_model.kneighbors(svd.transform(user_attraction_matrix.iloc[[user_idx]]), n_neighbors=5)
    
    similar_users = user_attraction_matrix.index[indices.flatten()[1:]]  # Exclude self
    user_ratings = user_attraction_matrix.loc[user_id]
    unseen_attractions = user_ratings[user_ratings == 0].index

    attraction_scores = {}
    for sim_user in similar_users:
        for attraction in unseen_attractions:
            attraction_scores[attraction] = attraction_scores.get(attraction, 0) + user_attraction_matrix.loc[sim_user, attraction]

    return sorted(attraction_scores, key=attraction_scores.get, reverse=True)[:num_recommendations]

#PREDICT THE RECOMMENDATION 
if st.button("🔮 Get Recommendations"):
    recommended = recommend_attractions(user_id)
    st.markdown("<h3> Recommended Attractions:</h3>", unsafe_allow_html=True)
    
    for i, attraction in enumerate(recommended, 1):
        st.markdown(f"<div class='recommendation-box'> ✨ {attraction}</div>", unsafe_allow_html=True)

#CREATE PREVIEW TABLE FOR USER FRIENDLY
st.subheader("📊 Preview of Tourism Data")
st.write("Below is a preview of the full tourism dataset used for predictions:")
st.dataframe(pd.read_csv(r"D:\1final ds\final_ds2.csv"))