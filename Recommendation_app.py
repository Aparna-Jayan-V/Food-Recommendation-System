
import streamlit as st
import pickle
import pandas as pd
import random

content_rec = pickle.load(open('/home/aparna/Downloads/content_recommendation.pkl','rb'))
content=list(content_rec)

recommended_item_ids = pickle.load(open('/home/aparna/Downloads/recommended_item_ids.pkl','rb'))
item_ids=list(recommended_item_ids)

data = pickle.load(open('/home/aparna/Downloads/data.pkl','rb'))
df_data=pd.DataFrame(data)

data_user = pickle.load(open('/home/aparna/Downloads/data_user.pkl','rb'))
df_data_user=pd.DataFrame(data_user)

def food_recommendation_system(user_id):
    recommendation_list=[]
    diet_name=[]
    to_recommend = random.sample((content + item_ids), 10)
    recommended_items = df_data[df_data["item_id"].isin(to_recommend)]
    for item in recommended_items.item_name:
        recommendation_list.append(item)
    for diet in recommended_items.diet_name:
        diet_name.append(diet)
    return recommendation_list,diet_name

st.title('Hybrid Food Recommender System')

selected_user_id=st.selectbox('Which user do you prefer?',df_data_user['user_id'].values)

if st.button('Foods Recommended'):
    
    item,diet=food_recommendation_system(selected_user_id)
    
    col1, col2 =st.columns(2)
    
    with col1:
        st.write(item[1])
    
        st.write(item[2])
   
        st.write(item[3])
    
        st.write(item[4])
    
        st.write(item[5])
        
        st.write(item[6])
        
    with col2:
        st.write(diet[1])
    
        st.write(diet[2])
   
        st.write(diet[3])
    
        st.write(diet[4])
    
        st.write(diet[5])
        
        st.write(diet[6])
    


