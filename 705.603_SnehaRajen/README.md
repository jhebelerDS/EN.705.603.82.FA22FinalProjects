# 705.603_SnehaRajen
JHU EP Creating AI Enabled Systems

Please go through each assignment # folder to find details on that coding assignment, the scripts and notebooks required, and other information.

## Assignment 3
For Assignment 3, the Amazon musical instrument review data on Kaggle was utilized. Using the summary column,  tokenization, stemming, and lemmatization were performed using the NLTK library and other NLP processing techniques. 
There is a docker image with tag assignment3. The python script will go through the preprocessing NLP steps as well as a sentiment analysis example. To run the image please run the command: docker run -it -v c:/output:/output srajen1/705.603_sneharajen:assignment3
visit docker page: https://hub.docker.com/repository/docker/srajen1/705.603_sneharajen/tags?page=1&ordering=last_updated

## Assignment 4
For Assignment 4, the Kaggle used car dataset was utilized. Processing and transforming the various categorical data was completed to be able to run a deep neural network.  

## Assignment 6
For Assignment 6, NoSQL applications were explored using MongoDB and Neo4j. Queries from a database where run and used to train a machine learning model.
  
## Assignment 10
For Assignment 10, AWS Sagemaker applications were explored. First, the AWS Sagemaker tutorial was completed. This involved operating AWS hosted jupyter notebooks, ingesting data (eg. S3), starting up a learning virtual machine, and deploying the model to a live web service. Then, a local and AWS implementation of Blackjack RL was created. The local RL was created first. Then this code was modified to use in the VM, training and deployment to a webservice also completed using AWS.
