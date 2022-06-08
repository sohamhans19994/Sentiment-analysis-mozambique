import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import spacy
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
import plotly.express as px
import requests
import argparse
import sys

from yaml import parse


class NewsSentimentAnalysis:
    def __init__(self):
        self.articles = []
        self.nlp = spacy.load('en_core_web_sm')

    def getURLData(self,url):
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            }
        req = requests.get(url, headers)
        soup = BeautifulSoup(req.content,'html.parser')
        return soup

    def fetchArticles(self):
        print("Fetching Articles")
        soup = self.getURLData("https://www.aljazeera.com/where/mozambique/")
        for a in soup.find_all('a',href=True,attrs={'class':'u-clickable-card__link'}):
            if re.match("/news/+",a['href']):
                if len(self.articles) == 10:
                    break
                title = a.get_text().replace("\xad","")
                article = {'url':'https://www.aljazeera.com'+a['href'],'title':title}
                self.articles.append(article)
        for article in self.articles:
            url = article['url']
            soup = self.getURLData(url)
            text = ""
            for div in soup.findAll('div',attrs={'class':'wysiwyg wysiwyg--all-content css-1ck9wyi'}):
                for p in div.find_all('p',recursive=False):
                    text += p.text
            article['text'] = text

    def saveAsJSON(self):
        print("saving JSON")
        with open("articles.json","w") as json_file:
            json.dump(self.articles,json_file,indent=4,ensure_ascii=False)
    
    def cleanArticles(self):
        print("Preprocessing article texts")
        for i in tqdm(range(int(len(self.articles)))):
            article = self.articles[i]
            raw = article['text']
            cleaned = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
            cleaned = re.sub('&gt;', "", cleaned)
            cleaned = re.sub('&#x27;', "'", cleaned)
            cleaned = re.sub('&quot;', '"', cleaned)
            cleaned = re.sub('&#x2F;', ' ', cleaned)
            cleaned = re.sub('<p>', ' ', cleaned)
            cleaned = re.sub('</i>', '', cleaned)
            cleaned = re.sub('&#62;', '', cleaned)
            cleaned = re.sub('<i>', ' ', cleaned)
            cleaned = re.sub("\n", ' ', cleaned)
            article['text'] = cleaned
    
    def extractSentences(self):
        for article in self.articles:
            sentences = []
            text = article['text']
            tokens = self.nlp(text)
            for sent in tokens.sents:
                sentences.append(str(sent))
            article['sentences'] = sentences
    
    def textblob_analysis(self):
        textblob_vals = []
        for article in self.articles:
            sentences = article['sentences']
            total_polarity = 0
            polarity_vals = []
            for s in sentences:
                blob = TextBlob(str(s))
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                polarity_vals.append(polarity)
                total_polarity += polarity
            article_vals = {'polarity_vals':polarity_vals}
            article_vals['polarity'] = total_polarity/len(sentences)
            textblob_vals.append(article_vals)
        return textblob_vals
    
    def flair_analysis(self):
        print("starting analysis")
        flair_classifier = TextClassifier.load('en-sentiment')
        flair_vals = []
        for article in self.articles:
            flair_val = {}
            values = []
            sentences = article['sentences']
            total_polarity = 0
            polarity_vals = []
            score = 0
            score_vals = []
            for s in sentences:
                sentence = Sentence(str(s))
                flair_classifier.predict(sentence)
                if len(sentence.labels) == 0:
                    continue
                value = sentence.labels[0].to_dict()['value'] 
                if value == 'POSITIVE':
                    result = sentence.labels[0].to_dict()['confidence']
                else:
                    result = -(sentence.labels[0].to_dict()['confidence'])
                values.append(result)
                score += result
            flair_val['polarity_vals'] = values
            flair_val['polarity'] = score/len(sentences)
            flair_vals.append(flair_val)
        return flair_vals
        
    def generate_plots(self,articles,polarity):
        fig = px.bar(y=polarity,title="Article Sentiment",color=articles,labels={'x':'article','y':'polarity'})
        fig.write_image("article_sentiment_graph.png")

    def main(self,textblob=False):
        self.fetchArticles()
        self.saveAsJSON()
        self.cleanArticles()
        self.extractSentences()
        article_names = [article['title'] for article in self.articles]
        if textblob:
            textblob_vals = self.textblob_analysis()
            textblob_polarities = [textblob_val['polarity'] for textblob_val in textblob_vals]
            self.generate_plots(article_names,textblob_polarities)
        else:
            flair_vals = self.flair_analysis()
            flair_polarities = [flair_val['polarity'] for flair_val in flair_vals]
            self.generate_plots(article_names,flair_polarities)


if __name__ == "__main__":
    newsSentiment = NewsSentimentAnalysis()
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    parser.add_argument('--textblob',action='store_true')
    if len(sys.argv)>1:
        if sys.argv[1] == '--textblob':
            newsSentiment.main(textblob=True)
        else:
            newsSentiment.main()
    else:
        newsSentiment.main()
