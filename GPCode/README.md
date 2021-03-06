![img0.jpg](.imgs/img0.jpg)

While the relationship between the sentiment of traders, and sentiment of the digital Twitter population, has been tested to have a relationship to the financial performance of industries, either through share price or financial performance, there has been limited studies of impact of sentiment of Critics review on the stock price of the publishing company.

The purpose of this study is to create a model that is predictive in nature of a publicly traded movie publishing company using sentiment analysis to help determine any directional shift of the share price. Is there a relation between comments made on by movie critic and the sudden change in price seen with the stocks? If so, does the comments have an immediate impact on the stocks, or is there a lag effect where comments made impact the stock a few days later? The project aims to analyze these questions and answer them in an objective manner.

### Methodology:

We have used Pandas heavily in data profiling, pre-processing and plotting.

![img7-real.jpg](./imgs/img7-real.jpg)

### Source of Data:

Critic Reviews Dataset :[Rotten Tomatoes movies and critic reviews dataset | Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

*Content:*

In the movies dataset each record represents a movie available on Rotten Tomatoes, with the URL used for the scraping, movie tile, description, genres, duration, director, actors, users' ratings, and critics' ratings.
In the critics dataset each record represents a critic review published on Rotten Tomatoes, with the URL used for the scraping, critic name, review publication, date, score, and content.

Publishing Company Stock Data: ALPACA API ( alpaca\_api.get\_bars )

Python Modules Installed:

![img4.jpg](./imgs/img4.jpg)

Other Modules Used:

- Statsmodels (pip install statsmodels)
- Textblob ( pip install textblob)
- Circlify (pip install circlify)
- Flair (pip install flair) May require Pytorch to be installed.
- Gensim (pip install gensim)

### Directory Structure

Code: Final Jupyter Notebooks. The code has been done for two publishing companies, Netflix and Disney. In the Kaggle data set, any row containing the publishing company name has been included.

data: Data Files used for Movie Critic comments.

Presentation: Deck for Project presentation

imgs: Images used in Readme and Presentation.

In addition to above team members may have their own branch directory.

### Data Profiling:

![MDP.PNG](./imgs/MDP.PNG)

**Stock Price Profiling:**

![SPDP.PNG](./imgs/SPDP.PNG)

![img6-nlp.jpg](./imgs/img6-nlp.jpg)

NLP Pipeline profiling:

![NLP.PNG](./imgs/NLP.PNG)

![img9-nlp.jpg](./imgs/img9-nlp.jpg)

NLP Score Comparison and Outcome:

![NLPScore.PNG](./imgs/NLPScore.PNG)

![NLPOutcome.PNG](./imgs/NLPOutcome.PNG)

**The Final Test:**

We join the movie comment and stock data using Pandas join on streaming\_release\_date (NFLX) and data of Stock price after normalizing the data using MinMaxScaler and plot the final data frame,

![Join.PNG](./imgs/Join.PNG)

The graph does show some promise on probable correlation.

**Granger Causality Test:**

![GC.PNG](./imgs/GC.PNG)

### **Conclusion:**

In the above case P-Value is indeed above 0.05 and hence we can assume that the critic comments do not impact stock value.

We should further test with different publishing company and much bigger dataset.