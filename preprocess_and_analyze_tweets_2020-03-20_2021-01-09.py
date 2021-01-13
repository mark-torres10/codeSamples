###### 

# Purpose of this script:
 
# Analysis of COVID tweets scraped, from March 2020 to early January 2021. 

#########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import wordcloud
from wordcloud import WordCloud
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import textblob
from textblob import TextBlob
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
import torch
from pprint import pprint
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pd.set_option('display.max_columns', None) # show all columns

IMPORT_DIR = "../../data/tweets/"
IMAGES_DIR = "../../media/images/"


tweets = pd.read_csv(IMPORT_DIR + "tweets_2020-03-20_2021-01-09_with_locations.csv")

# preprocessing
tweets = tweets.dropna(axis=0, subset=["cleaned_text", "cleaned_text_no_hashtags"])
tweets.drop_duplicates(inplace=True)
tweets["cleaned_text"] = tweets["cleaned_text"].apply(lambda x : ast.literal_eval(x))
tweets["hashtags"] = tweets["hashtags"].apply(lambda x : ast.literal_eval(x))
tweets["cleaned_text_no_hashtags"] = tweets["cleaned_text_no_hashtags"].apply(lambda x : ast.literal_eval(x))

# creating new date features
tweets["year_of_tweet"] = tweets["date_of_tweet"].apply(lambda x : x.split('-')[0])
tweets["year_month_tweet"] = tweets["year_of_tweet"] + "-" + str(tweets["month_of_tweet"])
tweets["year_month_tweet"] = ""
tweet_year_list = list(tweets["year_of_tweet"])
tweet_month_list = list(tweets["month_of_tweet"])
for idx in range(tweets.shape[0]):
    
    year = tweet_year_list[idx]
    month_int = int(tweet_month_list[idx])
    month = ""
    if month_int >= 10:
        month = str(month_int)
    else:
        month = '0' + str(month_int)
    
    year_month = year + '-' + month
    
    tweets["year_month_tweet"][idx] = year_month


##### exploratory data analysis

# how many tweets do we have, per state?
tweets["US_state"].value_counts()


##### Analysis of Hashtags

# How many hashtags do the tweets have?
plt.hist(tweets["hashtags_count"])
plt.show()

tweets["hashtags_count"].describe()


# What hashtags are most common?
all_hashtags = []
for lst in tweets["hashtags"]:
    try:
        if len(lst) > 0:
            for elem in lst:
                all_hashtags.append(elem)
    except Exception as e:
        continue


# visualize most common hashtags, using wordclouds
hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(all_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-03-20-2021-01-09.png")
plt.show()


# How does the distribution of hashtags vary across time?
march_hashtags = []
april_hashtags = []
may_hashtags = []
june_hashtags = []
july_hashtags = []
august_hashtags = []
september_hashtags = []
october_hashtags = []
november_hashtags = []
december_hashtags = []
january_hashtags = []

for month_num in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    hashtag_subset = tweets[tweets["month_of_tweet"]==month_num]["hashtags"]
    hashtag_lst = []
    for lst in hashtag_subset:
        try:
            if len(lst) > 0:
                for elem in lst:
                    hashtag_lst.append(elem)
        except Exception as e:
            continue

    if month_num == 1:
        january_hashtags = hashtag_lst
    elif month_num == 3:
        march_hashtags = hashtag_lst
    elif month_num == 4:
        april_hashtags = hashtag_lst
    elif month_num == 5:
        may_hashtags = hashtag_lst
    elif month_num == 6:
        june_hashtags = hashtag_lst
    elif month_num == 7:
        july_hashtags = hashtag_lst
    elif month_num == 8:
        august_hashtags = hashtag_lst
    elif month_num == 9:
        september_hashtags = hashtag_lst
    elif month_num == 10:
        october_hashtags = hashtag_lst
    elif month_num == 11:
        november_hashtags = hashtag_lst
    elif month_num == 12:
        december_hashtags = hashtag_lst

# wordcloud of COVID tweets for March 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(march_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-03.png")
plt.show()


# wordcloud of COVID tweets for April 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(april_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-04.png")
plt.show()

# wordcloud of COVID tweets for May 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(may_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-05.png")
plt.show()

# wordcloud of COVID tweets for June 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(june_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-06.png")
plt.show()

# wordcloud of COVID tweets for July 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(july_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-07.png")
plt.show()


# wordcloud of COVID tweets for August 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(august_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-08.png")
plt.show()

# wordcloud of COVID tweets for September 2020


hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(september_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-09.png")
plt.show()


# wordcloud of COVID tweets for October 2020

hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(october_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-10.png")
plt.show()


# wordcloud of COVID tweets for November 2020


hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(november_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-11.png")
plt.show()


# wordcloud of COVID tweets for December 2020


hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(december_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2020-12.png")
plt.show()

# wordcloud of COVID tweets for January 2021


hashtags_wc = WordCloud(width = 512,height = 512, collocations=False, colormap="viridis").generate(" ".join(january_hashtags))
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(hashtags_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.savefig(IMAGES_DIR + "covid_wordCloud_2021-01.png")
plt.show()


#### Topic Modelling

MODEL_DIR = "../../models/topic_models/"

# create dictionary, corpus for mental health tweets
dict_covid = corpora.Dictionary(tweets["cleaned_text"])
corpus_covid = [dict_covid.doc2bow(tweet) for tweet in tweets["cleaned_text"]]


# dump files
pickle.dump(corpus_covid, open(MODEL_DIR + 'corpus_covid_2020-03-20_2021-01-09.pkl', 'wb'))
dict_covid.save(MODEL_DIR + 'dictionary_covid_2020-03-20_2021-01-09.gensim')

# do LDA (try 12 topics)
NUM_TOPICS = 12
ldamodel_covid = gensim.models.ldamodel.LdaModel(corpus_covid,
                                                 num_topics = NUM_TOPICS, 
                                                 id2word=dict_covid, 
                                                 passes=15)

# save lda model
ldamodel_covid.save('lda_model_covid_2020-03-20_2021-01-09.gensim')

# display topics
topics = ldamodel_covid.print_topics(num_words=10)
for topic in topics:
    pprint(topic)

lda_display = pyLDAvis.gensim.prepare(ldamodel_covid, 
                                      corpus_covid, 
                                      dict_covid, sort_topics=False)
pyLDAvis.display(lda_display)


# Compute Perplexity
print('\nPerplexity: ', ldamodel_covid.log_perplexity(corpus_covid))  # a measure of how good the model is. lower the better.


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel_covid, texts=tweets["cleaned_text"], dictionary=dict_covid, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# get predominant topics for each tweet

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel_covid, corpus=corpus_covid, texts=tweets["cleaned_text"])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


# get predominant topic, assign as "label"

tweets["dominant_topic_label"] = df_dominant_topic["Dominant_Topic"]
tweets["dominant_topic_keywords"] = df_dominant_topic["Keywords"]

pd.crosstab(tweets["month_of_tweet"], tweets["dominant_topic_label"], margins = True, normalize = "index")


# Seems like there is some difference in the distribution of topics across time

# Let's see if there's a difference in the distribution of topics across locations (limiting our search to "CA", "NY", "TX", and "FL"
bool_state_arr = []
for state in tweets["US_state"]:
    if state in ["NY", "CA", "FL", "TX"]:
        bool_state_arr.append(True)
    else:
        bool_state_arr.append(False)

tweets_filtered = tweets[bool_state_arr]
pd.crosstab(tweets_filtered["US_state"], tweets_filtered["dominant_topic_label"], margins = True, normalize = "index")

# some difference in distribution of topics across state. Localized by state events, such as lockdowns?

#### How do keywords trend?


# #covid
has_covid = ["covid" in txt or "#covid" in txt for txt in tweets["cleaned_text"]]
covid_tweets = tweets[has_covid]
counts_covid_tweets = covid_tweets["year_month_tweet"].value_counts().sort_index()
counts_covid_tweets.plot()
plt.title(f"Count of tweets with the word 'covid'")
plt.show()


# #covid19
has_covid19 = ["covid19" in txt or "#covid19" in txt for txt in tweets["cleaned_text"]]
covid19_tweets = tweets[has_covid19]
counts_covid19_tweets = covid19_tweets["year_month_tweet"].value_counts().sort_index()
counts_covid19_tweets.plot()
plt.title(f"Count of tweets with the word 'covid19'")
plt.show()


# #lockdown
has_lockdown = ["lockdown" in txt or "#lockdown" in txt for txt in tweets["cleaned_text"]]
lockdown_tweets = tweets[has_lockdown]
counts_lockdown_tweets = lockdown_tweets["year_month_tweet"].value_counts().sort_index()
counts_lockdown_tweets.plot()
plt.title(f"Count of tweets with the word 'lockdown'")
plt.show()


# #newnormal
has_newnormal = ["newnormal" in txt or "#newnormal" in txt for txt in tweets["cleaned_text"]]
newnormal_tweets = tweets[has_newnormal]
counts_newnormal_tweets = newnormal_tweets["year_month_tweet"].value_counts().sort_index()
counts_newnormal_tweets.plot()
plt.title(f"Count of tweets with the word 'newnormal'")
plt.show()


# #vaccine
has_vaccine = ["vaccine" in txt or "#vaccine" in txt for txt in tweets["cleaned_text"]]
vaccine_tweets = tweets[has_vaccine]
counts_vaccine_tweets = vaccine_tweets["year_month_tweet"].value_counts().sort_index()
counts_vaccine_tweets.plot()
plt.title(f"Count of tweets with the word 'vaccine'")
plt.show()


# plot raw counts of words, by month, against each other
tweet_counts_by_word = pd.concat([counts_covid_tweets, counts_covid19_tweets, counts_lockdown_tweets, 
                                     counts_newnormal_tweets, counts_vaccine_tweets], axis=1)
tweet_counts_by_word.columns = ["covid", "covid19", "lockdown", "newnormal", "vaccine"]


tweet_counts_by_word.drop(index='', axis=0, inplace=True)

tweet_counts_by_word_normalized = pd.concat([normalized_counts_covid_tweets, normalized_counts_covid19_tweets, 
                                             normalized_counts_lockdown_tweets, normalized_counts_newnormal_tweets, 
                                             normalized_counts_vaccine_tweets], axis=1)

tweet_counts_by_word_normalized.columns = ["covid", "covid19", "lockdown", "newnormal", "vaccine"]

tweet_counts_by_word.plot()
plt.title(f"Count of tweets since March 2020, by word")
plt.savefig(IMAGES_DIR + "tweet_counts_by_word_2020-03-20_2021-01-09.png")
plt.show()

tweet_counts_by_word.drop(['covid19'], axis=1).plot()
plt.title(f"Count of tweets since March 2020, by word (excluding 'covid19')")
plt.savefig(IMAGES_DIR + "tweet_counts_by_word_excludeCovid19_2020-03-20_2021-01-09.png")
plt.show()

# plot (normalized) counts of words, by month, against each other
normalized_counts_covid_tweets = counts_covid_tweets / sum(counts_covid_tweets)
normalized_counts_covid19_tweets = counts_covid19_tweets / sum(counts_covid19_tweets)
normalized_counts_lockdown_tweets = counts_lockdown_tweets / sum(counts_lockdown_tweets)
normalized_counts_newnormal_tweets = counts_newnormal_tweets / sum(counts_newnormal_tweets)
normalized_counts_vaccine_tweets = counts_vaccine_tweets / sum(counts_vaccine_tweets)

tweet_counts_by_word_normalized.drop(index='', axis=0, inplace=True)

tweet_counts_by_word_normalized.plot()
plt.title(f"(Normalized) count of tweets since March 2020, by word")
plt.savefig(IMAGES_DIR + "normalized_tweet_counts_by_word_2020-03-20_2021-01-09.png")
plt.show()


#### How many tweets were US vs. non-US tweets?

tweets["is_US"] = tweets["US_state"].notnull()

plt.bar(x=["USA", "Not USA"], height=tweets["is_US"].value_counts().sort_values())
plt.title("Number of tweets, USA vs. outside of USA")
plt.savefig(IMAGES_DIR + "count_tweets_USA_notUSA.png")
plt.show()


#### What is the distribution of USA tweets, by state?

US_tweets = tweets[tweets["is_US"] == True]
US_states_count_top20 = US_tweets["US_state"].value_counts().head(20)
plt.bar(x=US_states_count_top20.index, height=US_states_count_top20, width=0.5)
plt.title("Tweet count, by US state")
plt.show()


#### Perform sentiment analysis on the tweets

# We'll perform sentiment analysis on the tweets. We'll use several ways to try and accomplish this:
# 
# 1. Pre-trained VADER models
# 2. Textblob
# 3. HuggingFace pre-trained models
# 

##### 
# VADER Model
# 
# VADER is a rule-based sentiment analysis model that is popular for social media sentiment analysis. 
# 
# Useful guide: https://predictivehacks.com/how-to-run-sentiment-analysis-in-python-using-vader/
# 
# The VADER library returns 4 values:
# 1. `pos`: the probability of the sentiment being positive
# 2. `neu`: the probability of the sentiment being neutral
# 3. `neg`: the probability of the sentiment being negative
# 4. `compound`: the normalized "single metric" of sentiment (score >= 0.5 --> positive, -0.05 < score < 0.05 --> neutral, score <-0.05 --> negative)
#####

analyzer = SentimentIntensityAnalyzer()

pos_scores = []
neu_scores = []
neg_scores = []
compound_scores = []

for idx, tokenized_text in enumerate(list(tweets["cleaned_text_no_hashtags"])):
    
    if idx % 1000 == 0:
        print(f"Finished with tweet {idx} out of {tweets.shape[0]}")
        
    combined_text = " ".join(tokenized_text)
    
    output_dict = analyzer.polarity_scores(combined_text)

    pos_scores.append(output_dict["pos"])
    neu_scores.append(output_dict["neu"])
    neg_scores.append(output_dict["neg"])
    compound_scores.append(output_dict["compound"])
    
    if idx <= 5:
        print(f"Here's an example tweet: {combined_text}")
        print(f"Here are the scores that it got: {output_dict}")

tweets["vader_pos_score"] = pos_scores
tweets["vader_neu_score"] = neu_scores
tweets["vader_neg_score"] = neg_scores
tweets["vader_compound_score"] = compound_scores


##### 
# TextBlob Sentiment Analysis
# 
# Useful resources for TextBlob:
# 
# 1. https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
# 2. https://stackabuse.com/sentiment-analysis-in-python-with-textblob/
# 
#####

polarity_scores = []
subjectivity_scores = []


# In[284]:


for idx, tokenized_text in enumerate(list(tweets["cleaned_text_no_hashtags"])):
    
    if idx % 1000 == 0:
        print(f"Finished with tweet {idx} out of {tweets.shape[0]}")
        
    combined_text = " ".join(tokenized_text)
    
    sentence = TextBlob(combined_text)

    sentiment=sentence.sentiment
    
    polarity_scores.append(sentiment.polarity)
    subjectivity_scores.append(sentiment.subjectivity)
    
    
    if idx <= 5:
        print(f"Here's an example tweet: {combined_text}")
        print(f"Here are the scores that it got: {sentiment}")


tweets["textblob_polarity_scores"] = polarity_scores
tweets["textblob_subjectivity_scores"] = subjectivity_scores



#### 
# DistilBERT
# 
# We'll be using DistilBERT, a condensed, smaller, faster version of BERT (trained on the losses of the original BERT model). 
# 
# It takes the original DistilBERT model and fine-tunes it on the SST-2 dataset.
# 
# Here's the HuggingFace link to their implementation: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# 
### 

tweets["cleaned_text_joined"] = tweets["cleaned_text_no_hashtags"].apply(lambda x : " ".join(x))


# Load pre-trained model from HuggingFace
sentiment_analyzer = pipeline("sentiment-analysis")

# run classifier
tweets["distilBERT_sentiment"] = ""

distilBERT_sentiments = []

for idx, tweet in enumerate(list(tweets["cleaned_text_joined"])):
    
    if idx % 1000 == 0:
        print(f"Getting the distilBERT sentiment for tweet {idx} out of {tweets.shape[0]}")
    
    distilBERT_sentiments.append(sentiment_analyzer(tweet))

tweets["distilBERT_sentiment"] = distilBERT_sentiments
tweets["distilBERT_sentiment_label"] = [x.get("label") for x in tweets["distilBERT_sentiment"]]
tweets["distilBERT_sentiment_score"] = [x.get("score") for x in tweets["distilBERT_sentiment"]]




