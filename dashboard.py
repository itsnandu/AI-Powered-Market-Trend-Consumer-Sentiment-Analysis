import streamlit as st
import pandas as pd
import plotly.express as px


# page config

st.set_page_config(
    page_title = "AI Market Trend & Consumer Sentiment Forecaster",
    layout="wide"
)

st.title("AI-Powered Market Trend & Consumer Sentiment Dashboard")
st.markdown("Consumer sentiment, topic trend, and social insights from reviews, news, and Reddit data")


# load data 

@st.cache_data
def load_data():
    reviews = pd.read_csv("data/category_wise_lda_output_with_topic_labels.csv")
    
    reddit = pd.read_excel("data/reddit_category_trend_data.xlsx")
    
    news = pd.read_csv("data/news_data_with_sentiment.csv")
    
    print(reddit.columns)
    if "review_date" in reviews.columns:
        reviews["review_date"]= pd.to_datetime(
            reviews["review_date"], errors="coerce"   #convert invalid date to NaT 
        )
        
    
    if "published_at" in news.columns:
        news["published_at"]= pd.to_datetime(
            news["published_at"], errors="coerce"   #convert invalid date to NaT 
        )
     
    if "created_date" in reddit.columns:
        reddit["created_date"]= pd.to_datetime(
            reddit["created_date"], errors="coerce"   #convert invalid date to NaT 
        )    
        
    return reviews, reddit, news   


reviews_df, reddit_df, news_df  = load_data()



# sidebar filters

st.sidebar.header("Filters")

source_filter = st.sidebar.multiselect(
    "Select Source",
    options=reviews_df["source"].unique(),
    default=reviews_df["source"].unique()
)


category_filter = st.sidebar.multiselect(
    "Select Category",
    options=reviews_df["category"].unique(),
    default=reviews_df["category"].unique()
)

filtered_reviews = reviews_df[
    (reviews_df["source"].isin(source_filter))&
    (reviews_df["category"].isin(category_filter))
]



# KPI Metrics

st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(filtered_reviews))
col2.metric("Positive %", round((filtered_reviews["sentiment_label"]=="Positive").mean()*100,1))
col3.metric("Negative %", round((filtered_reviews["sentiment_label"]=="Negative").mean()*100,1))
col4.metric("Neutral %", round((filtered_reviews["sentiment_label"]=="Neutral").mean()*100,1))


# sentiment distribution 

col1, col2 = st.columns(2)

with col1:
    sentiment_dist = reviews_df["sentiment_label"].value_counts().reset_index()
    sentiment_dist.columns=["Sentiment", "Count"]
    
    fig = px.pie(
        sentiment_dist, 
        names="Sentiment",
        values="Count",
        title="Overall Sentiment Distribution", 
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    category_sentiment = (
        reviews_df.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
    )
    
    fig = px.bar(
        category_sentiment, 
        x="category",
        y="count",
        color="sentiment_label",
        title="Category-wise Sentiment Comparison",
        barmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    
# sentiment trend over time 

st.subheader("Sentiment Trend Over Time")


sentiment_trand= (
    filtered_reviews.groupby([pd.Grouper(key="review_date", freq="W"), "sentiment_label"])
    .size()
    .reset_index(name="count")
)    


fig_trend = px.line(
    sentiment_trand,
    x="review_date",
    y="count",
    color = "sentiment_label",
    title="Weekly Sentiment Trend"
)

st.plotly_chart(fig_trend, use_container_width=True)