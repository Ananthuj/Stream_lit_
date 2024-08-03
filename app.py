import streamlit as st
import pandas as pd

import pickle
model_file=open("modeloutput.pkl","rb")
model = pickle.load(model_file)

scaler_file = open("scaler.pkl", "rb")
scaler = pickle.load(scaler_file)

st.header("This is an application to predit the total Litres of pure Alcohol.")

continent=st.selectbox("Continent",['Asia','Europe','Africa','North America','South America','Oceania'])
beerserving=st.number_input("Beer Serving:")
spiritserving=st.number_input("Spirit Serving:")
wineserving=st.number_input("Wine Serving:")

if continent!=None and continent=="Asia":
    continent=0
elif continent!=None and continent=="Europe":
    continent=1
elif continent!=None and continent=="Africa":
    continent=2
elif continent!=None and continent=="North America":
    continent=3
elif continent!=None and continent=="South America":
    continent=4
elif continent!=None and continent=="Oceania":
    continent=5

if st.button("Make prediction") and continent!=None and beerserving!=None and spiritserving!=None and wineserving!=None:
    import numpy as np
    input_features=np.array([[continent,int(beerserving),int(spiritserving),int(wineserving)]])
    input_df = pd.DataFrame(input_features, columns=['continent', 'beer_servings', 'spirit_servings', 'wine_servings'])
    
    # Apply scaling
    input_df[['beer_servings', 'spirit_servings', 'wine_servings']] = scaler.transform(input_df[['beer_servings', 'spirit_servings', 'wine_servings']])
    
    # Extract scaled features as a numpy array
    input_features_scaled = input_df.values

    prediction = model.predict(input_features_scaled)
    prediction = prediction.tolist()
    st.write(f"Predicted total litres of pure alcohol: {prediction[0]}")
    st.balloons()


