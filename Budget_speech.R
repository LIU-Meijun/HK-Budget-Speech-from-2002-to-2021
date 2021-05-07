# 0. installing and loading packages ----- 

# for basic statistics 
library(foreign)
# install.packages("psych")
library(psych) 
#install.packages("NLP")
#install.packages("tm") 
library(NLP) 
library(tm)
#install.packages("ldatuning")
#install.packages("slam")
library(ldatuning)
library(slam)
#install.packages("devtools")
#devtools::install_github("nikita-moor/ldatuning")
# for calculating cosine similarity 
#install.packages("lsa")
library(lsa)

# 1. import data ----- 
# this step starts with exporting the data from external sources AFTER the data collection procedure 

budget_tweet_01 <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)  # file.choose()
# select the variables 
vars <- c("No.", "Year", "Title", "Content")
budget_tweet_01_3 <- subset(budget_tweet_01, select = vars) 

dim(budget_tweet_01_3)
names(budget_tweet_01_3)
head(budget_tweet_01_3) 

# 2. Inspecting the dataset and variables (with EDAV)  ----- 

# checklist of inspecting variables (an incomplete list)
names(budget_tweet_01_3)
names(budget_tweet_01_3)[names(budget_tweet_01_3) == 'No.'] <- 'doc_id'

class(budget_tweet_01_3$Content) # character 
names(budget_tweet_01_3)[names(budget_tweet_01_3) == 'Content'] <- 'text'

# 3. unsupervised machine learning for text analytics -----
# preparing the dataset for liberal media (Huffinton Post and BuzzFeed News) and conservative media (Fox)
budget_tweet_content <- budget_tweet_01_3
budget_tweet_title <- budget_tweet_01_3


## 3.1. Data cleaning for text data  -----

# to create a VCorpus object 
# source type: for a vector with “character” class one should use VectorSource(). for a dataframe, one should use DataframeSource() and name the two columns as "doc_id" and "text" 
#VCorpus() #creates a volatile corpus, which is the data type used by the tm package for text mining 
tweets_con_docs <- subset(budget_tweet_content, select = c("doc_id", "text"))
tweets_con_VCorpus <- VCorpus(DataframeSource(tweets_con_docs)) 
head(tweets_con_docs)
# to inspect the contents 
tweets_con_VCorpus[[1]]$content

# to convert to all lower cases
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_con_VCorpus[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, removeWords, stopwords("english"))
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, removeWords, c("hong","kong","billion","million","per","cent","year","rate","years","kongs","will","shall","last","total","also","now","can","must"))

# to remove extra whitespaces 
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, stripWhitespace) 

# to remove punctuations 
tweets_con_VCorpus <- tm_map(tweets_con_VCorpus, removePunctuation)
tweets_con_VCorpus[[1]]$content


# 3.2. TF and TF-IDF -----

# the *bags of words* model
# the bag of words model is a common way to represent documents in matrix form based on their term frequencies (TFs). 
# We can construct a document-term matrix (DTM), where n is the number of documents, and t is the number of unique terms. Each column in the DTM represents a unique term, the (i,j)th cell represents how many of term j are present in document i.

# converting to Document-term matrix (TDM)
tweets_con_dtm <- DocumentTermMatrix(tweets_con_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_con_dtm
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_con_dtm) # a sample of the matrix 

# TF
term_freq_con <- colSums(as.matrix(tweets_con_dtm)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_con)), decreasing=TRUE)), file="budget_tf.csv")

# TF-IDF
tweets_con_dtm_tfidf <- DocumentTermMatrix(tweets_con_VCorpus, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_con_dtm_tfidf) 
tweets_con_dtm_tfidf2 = removeSparseTerms(tweets_con_dtm_tfidf, 0.99)
print(tweets_con_dtm_tfidf2) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_con_dtm_tfidf2)), decreasing=TRUE)), file="budget_tfidf.csv")

# 3.3. topic modeling with LDA-----
#install.packages("topicmodels")
library(topicmodels)
#install.packages("ldatuning")
#install.packages("slam")
library(ldatuning)
library(slam)
#install.packages("devtools")
#devtools::install_github("nikita-moor/ldatuning")
# clean the empty (non-zero entry) 
rowTotals_con <- apply(tweets_con_dtm , 1, sum) #Find the sum of words in each Document
tweets_con_dtm_nonzero <- tweets_con_dtm[rowTotals_con> 0, ]

result <- FindTopicsNumber(
  tweets_con_dtm_nonzero,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
) # cannot plot Griffiths2004, but the other three work 
FindTopicsNumber_plot(result)
# hence, the optimized number of topics would be 4 as the trade-off 

# after finding "the optimal-K" topics, then redo the above analysis
# k10 - 4 topics, 10 term 
tweets_con_dtm_10topics <- LDA(tweets_con_dtm_nonzero, k = 4, method = "Gibbs", control = list(iter=2000, seed = 2000)) # find k topics
tweets_con_dtm_10topics_10words <- terms(tweets_con_dtm_10topics, 20) # get top 10 words of every topic
(tweets_con_dtm_10topics_10words <- apply(tweets_con_dtm_10topics_10words, MARGIN = 2, paste, collapse = ", "))  # show the results immediately, if having a ()  parenthesis 

tweets_con_dtm_10topics <- LDA(tweets_con_dtm_nonzero, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) # find k topics
tweets_con_dtm_10topics_10words <- terms(tweets_con_dtm_10topics, 20) # get top 10 words of every topic
(tweets_con_dtm_10topics_10words <- apply(tweets_con_dtm_10topics_10words, MARGIN = 2, paste, collapse = ", "))  
library(LDAvis)
json <- createJSON(phi = exp(tweets_con_dtm_10topics@beta), #topic-terms distribution matrix
                   theta = tweets_con_dtm_10topics@gamma, #document-topic distribution matrix
                   vocab = tweets_con_dtm_10topics@terms, #list of terms
                   doc.length = rowSums(as.matrix(tweets_con_dtm)), 
                   term.frequency = colSums(as.matrix(tweets_con_dtm))) #term frequency
#use servi to create html file，and save with out.dir
install.packages("servr")
serVis(json, out.dir = './vis', open.browser = T)

tweets_con_dtm_10topics <- LDA(tweets_con_dtm_nonzero, k = 6, method = "Gibbs", control = list(iter=2000, seed = 2000)) # find k topics
tweets_con_dtm_10topics_10words <- terms(tweets_con_dtm_10topics, 20) # get top 10 words of every topic
(tweets_con_dtm_10topics_10words <- apply(tweets_con_dtm_10topics_10words, MARGIN = 2, paste, collapse = ", "))  


# 3.4 finding document similarity -----
# Cosine similarity - based on a document-term matrix

#install.packages("lsa")
library(lsa)

vec1 = c( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
vec2 = c( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 )
cosine(vec1,vec2) 
vec3 = c( 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0 )
matrix = cbind(vec1,vec2, vec3)
cosine(matrix)

# document-term matrix for the LSA package 
# cosine() calculates a similarity matrix between all column vectors of a matrix x. 
# This matrix might be a *document-term matrix*, so columns would be expected to be documents and rows to be terms.
inspect(tweets_con_dtm)

tweets_con_mat <- as.matrix(tweets_con_dtm) 
tweets_con_mat_2 <- t(as.matrix(tweets_con_dtm)) # t means to transpose 


# https://rdrr.io/cran/lsa/man/cosine.html
cosine_2 <- cosine(tweets_con_mat_2)
inspect(cosine_2)
write.csv(cosine_2, "cosine_simi.csv")

# 4. wordcloud
library("ggplot2")
library("tm")
library("wordcloud")
library("RColorBrewer")
v <- sort(colSums(as.matrix(tweets_con_dtm_tfidf2)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 50)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order= FALSE, rot.per=0.0,
          colors=brewer.pal(8,"Dark2"))

# wordcloud2
wordcloud2(d, color = 'random-light')

#For title
lq.v <- VectorSource(budget_tweet_01_3$Title)
lq.c <- SimpleCorpus(lq.v)
inspect(lq.c)
lq.c.p <- tm_map(lq.c, content_transformer(tolower))
lq.c.p <- tm_map(lq.c.p, removeNumbers)
lq.c.p <- tm_map(lq.c.p, removeWords, stopwords("english"))
lq.c.p <- tm_map(lq.c.p, removeWords, c("lgbtq"))
lq.c.p <- tm_map(lq.c.p, removePunctuation)
lq.c.p <- tm_map(lq.c.p, stripWhitespace)
inspect(lq.c.p)
dtm <- TermDocumentMatrix(lq.c.p)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
wordcloud2(d, color = 'random-light')
###################################################### the end of the codes ---

