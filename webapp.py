import pickle
import streamlit as st
import numpy as np
import pandas as pd
import sklearn.metrics  as sk_metrics
from PIL import Image
 
# loading the trained model & Dataset
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
nada_dataset=pd.read_csv("Nada Dataset")
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 

def scaleDown(new_input):
    oldMax=12
    oldMin=3
    newMax=1
    newMin=0.11

    oldRange = oldMax - oldMin
    newRange = newMax - newMin
    new_values=[]
    for i in new_input:
        oldValue=np.sum(i)
        newValue = ((oldValue - oldMin) * newRange / oldRange) + newMin
        new_values.append(np.round(newValue,decimals=2))

    return [new_values]

def prediction(q1_1,q1_2,q1_3,q2_1,q2_2,q2_3,q3_1,q3_2,q3_3,q4_1,q4_2,q4_3,Gender,Sexualorientation,Preferred_age):   

    # Preprocessing
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1

    if Sexualorientation == "Heterosexual":
        Sexualorientation = 0
    elif Sexualorientation == "Homosexual":
        Sexualorientation = 1
    else:
        Sexualorientation = 2

    preferred_age=(np.arange(Preferred_age[0],Preferred_age[1],1))
        

 
    # Scaling   
    new_input=np.array([[q1_1,q1_2,q1_3],[q2_1,q2_2,q2_3],[q3_1,q3_2,q3_3],[q4_1,q4_2,q4_3]])
    new_profile=scaleDown(new_input)
 
    # Making predictions 
    prediction = classifier.predict( 
        [[new_profile[0][0], new_profile[0][1], new_profile[0][2], new_profile[0][3]]])


    # Filtering data, perform cosine similarity and sort the value based on similarity score
    df_result=nada_dataset[nada_dataset["clusters"]==prediction[0]]

    df_result["similarity_score"]=sk_metrics.pairwise.cosine_similarity(X = df_result.drop(["clusters","Name","Gender","Age","Sexual orientation","Unnamed: 0"], 1), Y=new_profile, dense_output=True)

    if Sexualorientation == 0:
        if Gender == 0:
            filtered_result=df_result.loc[(df_result['Gender'] == 1) & (df_result["Sexual orientation"]==0) & (df_result['Age'].isin(preferred_age))]
        elif Gender == 1:
            filtered_result=df_result.loc[(df_result['Gender'] == 0) & (df_result["Sexual orientation"]==0) & (df_result['Age'].isin(preferred_age))]
    elif Sexualorientation == 1:
        if Gender == 0:
            filtered_result=df_result.loc[(df_result['Gender'] == 0) & (df_result["Sexual orientation"]==1) & (df_result['Age'].isin(preferred_age))]
        elif Gender == 1:
            filtered_result=df_result.loc[(df_result['Gender'] == 1) & (df_result["Sexual orientation"]==1) & (df_result['Age'].isin(preferred_age))]
    elif Sexualorientation == 2:
        filtered_result=df_result.loc[(df_result["Sexual orientation"]==2) & (df_result['Age'].isin(preferred_age))]

    final_result = filtered_result.sort_values(by=["similarity_score"],ascending=False).head(5)
    final_result = final_result.drop(["Unnamed: 0"], 1)

    # Changing 0 and 1 value to Male and Female for Showing purpose
    final_result.rename(columns = {'Extroversion':'Extraversion'}, inplace = True)
    final_result['Gender'] = final_result['Gender'].replace([0],'Male')
    final_result['Gender'] = final_result['Gender'].replace([1],'Female')

    #Return 
    return st.write(final_result)
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    st.title("Nada AI-MatchMaker")
    image = Image.open('pict.jpg')

    st.image(image, use_column_width=True)
    st.write("Using Machine Learning to Find the Top Dating Profiles for you")
      
    
      
    # following lines create boxes in which user can enter data required to make prediction

    Gender = st.selectbox('Genre',("Male","Female"))
    Sexualorientation = st.selectbox('Sexual Orientation',("Heterosexual","Homosexual","Bisexual"))
    Preferred_age = st.slider('Select preferred age', 20, 60, (20, 60), 1)
    # Openness 
    st.subheader("Openness")
    q1_1 = st.slider("How adventurous and creative are you?", 1, 4)
    q1_2 = st.slider("How open are you to new and different ideas?", 1, 4) 
    q1_3 = st.slider("How happy are you to consider abstract principles?", 1, 4)

    # Agreeableness 
    st.subheader("Agreeableness")
    q2_1 = st.slider("How much do you like in corporate with other people", 1, 4)
    q2_2 = st.slider("How considerate and kind are you to others?", 1, 4)
    q2_3 = st.slider("How reliable worker do you think you are?", 1, 4)

    # Conscientiousness
    st.subheader("Conscientiousness") 
    q3_1 = st.slider("How likely are you to become disorganized?", 1, 4)
    q3_2 = st.slider("How much do you like in routine work?", 1, 4)
    q3_3 = st.slider("How much do you like in making plans and following through with them?", 1, 4)

    # Extroversion 
    st.subheader("Extraversion")
    q4_1 = st.slider("How much do you like being the center of attention?", 1, 4)
    q4_2 = st.slider("How much do you see yourself as a reserved person", 1, 4)
    q4_3 = st.slider("How much do you see yourself as someone who gets energy from social gatherings", 1, 4)


    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Find your couple!"): 
        result = prediction(q1_1,q1_2,q1_3,q2_1,q2_2,q2_3,q3_1,q3_2,q3_3,q4_1,q4_2,q4_3,Gender,Sexualorientation,Preferred_age) 
        st.balloons()
        st.success("Found your Top 5 Most Similar Personality traits!")    
     
if __name__=='__main__': 
    main()