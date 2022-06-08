# Summary

## Code documentation

1) Used a python virtual enviroment to install required packages, requirements are mentioned in requirements.txt <br />
2) The code is structured as a class named NewsSentimentAnalysis, with separate functions for carrying out different phases of the task <br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• The first function fetchArticles gets the data from the given url (https://www.aljazeera.com/where/mozambique/), and extracts the top &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10 news articles and their texts.<br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• The saveAsJSON function stores the collected data in JSON form. <br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• The article texts are then cleaned in the cleanArticles function. <br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Sentences are extracted from the articles in the extractSentences function. <br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Two separate functions exist for analysing the sentiment of the sentences in the articles. <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. The textblob_analysis function uses the textblob library to return the polarity of the sentences, the average polarity of all the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sentences in an article is returned as the polarity of the article. <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. The flair_analysis function uses the flair library and their pretrained models to get the sentiment of each sentence and it’s &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;confidence score. The overall sentiment of each article is retrieved by averaging over the scores of each sentence. <br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• The main function inside the class calls all the functions mentioned above in the right order. <br />

3) Running the code <br />
    • To run the code simply use the command <br />
	    `python3 sentiment_analysis.py` <br />
	    This command returns the analysis with flair <br />
	    CPU run time (user + sys): 9.484s <br />

    • In case you want to see the results for the textblob analysis, please run <br />
      `python3 sentiment_analysis.py --textblob` <br />
      CPU run time (user + sys): 5.757s<br />


## Results<br />

I have used two approaches to analyse the sentiments of the articles, and compared the results. The first method uses Textblob which is a lexicon-based approach that tries to figure out the sentiment of a sentence by checking the semantic orientation and intensity of every word in the sentence. The drawback with this approach is that it is unable to actually understand the relationship between the words in a sentence and the context and struggles with complicated sentences that have a lot of neutral words. The second approach, Flair, uses neural network based transformer models such as BERT. This allows it to understand the context in each sentence and then analyse the sentiment. The only drawback is that it usually takes much longer to analyse the sentences. <br />

Most of the articles retrieved have a negative sentiment, and the analysis using flair captures this trend pretty accurately. Whereas, the textblob approach ends up giving an almost neutral  or slightly sentiment to all the articles. Hence, using flair gives us a better accuracy. <br />
