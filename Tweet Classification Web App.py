#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:24:51 2022

@author: fbarde
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
import streamlit_wordcloud as wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px

from plotly.subplots import make_subplots
import re
import time
import os


#loaded model
loaded_model = pickle.load(open("TM12_1.0_LogReg_model.pkl", "rb"))

#load dataset

raw = pd.read_csv("train.csv")
#creating a function for prediction

def tweet_prediction(tweet_input):
    
    #tweet_input = []
    
    #tweet_input = [input('enter a text: ')]
    prediction=loaded_model.predict(tweet_input)

    if prediction < 0:
        return "Negative"
    elif prediction == 0:
        return "Neutral"
    else:
        return "Positive"

#creating WordCloud
stopwords = set(STOPWORDS)

extra_stopwords = ["The", "It", "it", "in", "In", "wh","rt"]

def processed_text(message):
  message = re.sub("https?:\/\/\S+", "", message)  # replacing url with domain name
  message = re.sub("#[A-Za-z0–9]+", " ", message)  # removing #mentions
  message = re.sub("#", " ", message)  # removing hash tag
  message = re.sub("\n", " ", message)  # removing \n
  message = re.sub("@[A-Za-z0–9]+", "", message)  # removing @mentions
  message = re.sub("RT", "", message)  # removing RT
  message = re.sub("^[a-zA-Z]{1,2}$", "", message)  # removing 1-2 char long words
  message = re.sub("\w*\d\w*", "", message)  

  for word in extra_stopwords:
        message = message.replace(word, "")
        
        message = message.lower()
    # will split and join the words
        message=' '.join(message.split())
  return message

#creating sentiment ratio
def getPolarity(message):
    sentiment_polarity = TextBlob(message).sentiment.polarity
    return sentiment_polarity
#convert polarity to sentiment
def getAnalysis(polarity_score):
    if polarity_score < 0:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    else:
        return "Positive"
    
#convert sentiment 
######################################################
    
def main():
    
    #Page Title
    st.set_page_config(page_title="TRIDENT AI", page_icon='dragon')
    #title for streamlit
    st.title('Tweet Classification Prediction')
    
    ######### body container ############
    from PIL import Image
    with st.sidebar.container():
        image = Image.open("icon.png")
        st.image(image, use_column_width=True)

    
    ######### Side Bar #################
    options = ["Prediction", "Information", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the "Information" page
    if selection == "Information":
        
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):
            st.write(raw[['sentiment', 'message']])
            
            def plot_sentiment(sentiment):
                df = raw[raw['sentiment']==sentiment]
                count = raw['sentiment'].value_counts()
                count = pd.DataFrame({'Sentiment':count.index, 'message':count.values.flatten()})
                return count
            
            st.subheader("Total Number of Tweets For Each Sentiment Group")
            each_sentiment = st.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
            tweet_sentiment_count = raw.groupby('sentiment')['sentiment'].count().sort_values(ascending=False)
            tweet_sentiment_count = pd.DataFrame({'sentiment':tweet_sentiment_count.index, 'message':tweet_sentiment_count.values.flatten()})
            if not st.checkbox("Close", True, key='2'):
                if each_sentiment == 'Bar plot':
                    st.subheader("Total number of tweets for each Sentiment")
                    fig_1 = px.bar(tweet_sentiment_count, x='sentiment', y='message', color='message', height=500)
                    st.plotly_chart(fig_1)
                if each_sentiment == 'Pie chart':
                    st.subheader("Total number of tweets for each Sentiment")
                    fig_2 = px.pie(tweet_sentiment_count, values='message', names='sentiment')
                    st.plotly_chart(fig_2)
            
            choice = st.multiselect('Select Sentiment', ('Anti','Neutral','Pro','News'), key=0)
            if len(choice) > 0:
                breakdown_type = st.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
                fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
                if breakdown_type == 'Bar plot':
                    for i in range(1):
                        for j in range(len(choice)):
                            fig_3.add_trace(
                        go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).message, showlegend=False),
                        row=i+1, col=j+1
                    )
                    fig_3.update_layout(height=600, width=800)
                    st.plotly_chart(fig_3)
                else:
                    fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
                    for i in range(1):
                        for j in range(len(choice)):
                            fig_3.add_trace(
                        go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).message, showlegend=True),
                        i+1, j+1
                    )
                    fig_3.update_layout(height=600, width=800)
                    st.plotly_chart(fig_3)
            
            
                        
                
    
        
            
                
        
        
        
    
    # Building out the predication page
    if selection == "Prediction":
         
         
         
         #input data field
         message = st.text_input("Type a Message")
         
         #code for prediction
         tweet = ''
         #button for classification
         if st.button("Classify"):
             st.info("Prediction with ML Models")
             tweet = tweet_prediction([message])
             st.success(tweet)
             
             #word cloud
             st.info("Word Cloud")
             
             wordcld = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(message)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             plt.imshow(wordcld)
             plt.xticks([])
             plt.yticks([])
             st.pyplot()
             
             #Sentiment Ratio
             st.info("Sentiment Ratio")
             #message_sentiment = pd.DataFrame(["message"]).apply(processed_text)
             #message_sentiment = message_sentiment["message"].apply(processed_text)
             
             
             
             
             
             
         #st.success(selection)
    # Building out the "Information" page
    if selection == "About Us":   
        st.info(
            "Discover Trident AI :dragon:"
        )
        st.markdown(
            """
                                    Welcome to Trident AI! 
            
            We're about Climate Change sustainability management in developing countries and around the globe. 
            
            Trident AI was built with long-term change management in mind. 
            
            Trident AI climate change solutions and technology help companies around the world achieve science-based sustainability goals, minimize climate impact, and eventually achieve net-zero and carbon neutrality. 
            
            Trident AI climate change solutions enable sustainability leaders and professionals to develop and implement sustainability strategies using sophisticated net-zero routes to tackle climate threats in a constantly changing climate.

"""
)
        st.info("Meet The Team!")
        from PIL import Image
        with st.container():
            image = Image.open("team.png")
            st.image(image, use_column_width=True)
    
   
    
    
    
    
    
    
    
    
        
        
    
 
    
    
    
    
   
  
  
   
    

if __name__ == '__main__':
    main()

    