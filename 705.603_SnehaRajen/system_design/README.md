# Telecom Customer Churn and Marketing Generated Content

## Introduction

**Background**: For my system design project, I wanted to pair GPT-3 with a churn prediction algorithm for a telecom company. The company wants to proactively identify customers who are likely to churn and stop purchasing. Once a customer is identified to churn, GPT-3 can send them an informational email on the company's product offerings to entice the customer and get them to retain. 

**Modeling**: There are several applications of models for this project.  
Step 1: Creating a binary classification model to help predict customer churn.
Step 2: Using the model's prediction, proactively generate an email. The email will entice the customer to stay and ideally purchase again. I will use GPT-3 to help reduce workload on campaign staff by proactively writing a repository of emails based on the company's product IT offerings. The campaign staff can then pick from a collection of premade blurbs to incorporate into their emails.

### 6Ds
**Decomposition**:
Reduce workload: In marketing campaigns there is traditionally an entire team of people for each type of campaign that are behind the scenes drafting up different emails customers receive. Using GPT-3 I hope to reduce the overall workload and process of writing multiple different email drafts (that all sound essentially the same). This will free up the campaign team to do other more creative work.  

Speed up the process: while it takes a long time and a lot of energy to draft up different emails, it takes even longer to actually get the email approved by legal. This is arguably the slowest part of the entire process. I hope by using GPT-3 I can generate several emails for various ocassions. i.e. xmas holiday marketing email, halloween, labor day sale, etc. that way multiple emails can be sent to legal at once, early, and the overall process of creating emails to getting legal approval is sped up.  

Achieve new insight: In addition to using GPT-3 the other big question I hope to answer is looking at and predicting customer churn. By proactively identifying customers we believe will churn in the future, we can use GPT-3 to send them a personalized email based on previous product interactions with our brand.  

**Domain Expertise**:
In general we need to be mindful of marketing 'donts' that come up. For example we can not create offers or emails that discriminate based on age, race, religion, etc. It is criticially important to check the text GPT-3 is producing to make sure these biases are minimized. It is also important to have someone knowledgeable in IT read the email blurbs and make sure it is logical.  

**Data**: The data is telecommunications data provided from kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn    
Each row represents a customer, each column contains customer’s attributes described below:  
Customers who left within the last month – Churn  
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies  
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges  
Demographic info about customers – gender, age range, and if they have partners and dependents  

**Design**: the AI solution used was a neural network with binary cross entropy loss function. Hyperparameter tuning was completed and integrated into a pipeline. The entire process is stored in a docker image. I have also used this to create an azure AI infrastructure with connectivity to powerbi dashboard.  

**Diagnosis**: For checking for bias, I am predominantly concerned about GPT-3's outputs as these are based on human provided examples and suggestions. I hoped to minimize this by having GPT only 'tell me facts' about various IT offerings. By asking for fun informative product offering descriptions which can then be included in emails and marketing communications, I hope to avoid subjective human bias.   

**Deployment**: For a customer churn model, concept drift definitely needs to be monitored as new influences (products, services, experiences, external factors etc) all play a role.For GPT-3 text generation I see less of a need for continuous monitoring as it was designed to generate a bunch of email blurbs to be used and vetted by legal in one sitting - say once a quarter - so more prevalent monitoring shouldnt be required unlike the actual churn model. Continuous monitoring will need to be done for long term use of the churn model which will likely require retaining of the model when new data becomes present, or a lot of time has passed since the initial training.  

**System Development Pipeline**:  
Using github and docker.  
Hosting Environment - MS Azure  
Storage - MS Azure's sql database

### Create an Azure Resource Group for the solution
![resource group](https://user-images.githubusercontent.com/113488887/207524515-7668addd-2c34-4337-99bd-f80e5b173904.png)

### Create an Azure ML Workspace
Details for ML workspace "sysdesign":  
![ml environment](https://user-images.githubusercontent.com/113488887/207530481-35249241-650e-4f20-8fb2-d84a88d9b95a.png)

Data was uploaded via Azure Storage Account - blob storage and then utilized in autoML. In AutoML Classification was selected and the 'churn' column was picked as the target.
Here we can see the many models it was compared against
![automl](https://user-images.githubusercontent.com/113488887/207524479-aa5d374b-e330-4b4e-bdc3-ddd951c1edee.png)

Below is the data cleaning process taken:  
![dataclean](https://user-images.githubusercontent.com/113488887/207530828-ad0b523a-b66e-41b3-b1e6-a8e7ce6e211d.png)

We can also understand the top variables and feature importance:  
![feature importance](https://user-images.githubusercontent.com/113488887/207530943-dd8f7f76-5e16-46ce-9bd9-37126b61c579.png)

We have detailed views of model performance metrics:  
![metrics](https://user-images.githubusercontent.com/113488887/207530999-5381516d-da55-4c04-a38e-10c5cc7a051e.png)
![model performance](https://user-images.githubusercontent.com/113488887/207531021-c1b48e4c-037a-46f4-b158-d1a7ed886e2b.png)

### Deploy best model to endpoint
Deploy the model via endpoint: following this tutorial: https://learn.microsoft.com/en-us/azure/machine-learning/v1/tutorial-power-bi-custom-model

![endpoint](https://user-images.githubusercontent.com/113488887/207531369-a33c173b-48d7-4de8-985a-29792a6ffe66.png)

## Connect the model to powerbi
This will let stakeholder view a BI solution and understand basic trends on their data. It will also let them input data to get predictions on. Follow this tutorial: https://learn.microsoft.com/en-us/power-bi/connect-data/service-aml-integrate?context=azure%2Fmachine-learning%2Fcontext%2Fml-context
![pbi](https://user-images.githubusercontent.com/113488887/207532048-9f427d8a-f126-4038-a4fa-feba877c000c.png)
![pbi connecting](https://user-images.githubusercontent.com/113488887/207532023-ca746796-afbe-4c30-a557-edc8b0bd71c7.png)
![dash](https://user-images.githubusercontent.com/113488887/207531992-7a7fd6bb-d9bf-437c-aed2-c4d7980d5443.png)

Now we have succesfully deployed our trained model to an endpoint, used that to connect to powerbi, and can even make single point predictions within powerbi!

**Model Selection**: I opted for AutoML to identify the best parameters and model. The winning combo was XGBoost Classifier with standard scaling. There are only 11 nulls, instead of dropping the model imputed using mean. Looking at performance metrics we can see the accuracy ~81%. For text generation models, there is really one big winner: GPT-3. This was a challenging architecture to learn as GPT-3 isn't your typical model but rather it has 175 B Parameters, 96 attention layers, and a 3.2 M batch size and is a modified/amplified transformer architecture. While I did not train GPT-3 from scratch I did use it via transfer learning and was able to generate text.

**Analysis**: I enjoyed using Azure's infrastructure to set this up. Although the challenge was getting GPT-3 to cooperate. Because Azure has their own inhouse GPT-3 it seems to block openai.apikey that I used to run GPT-3 locally. Azure has a very extensive policy regarding openAI tools and responsible use. I applied for access some time ago and was actually denied as a student because they seem to be giving access to researchers. Was not anticipating this when I started my project. In fact, there were several challenges with the project design I initially had, specifically with respect to getting developer API access approval. Initially I had planned to use Azure Event Hub to stream tweets in real time to identify topics which could then be fed into GPT-3 to generate promotional content. I had initially requested access to developer Twitter API weeks ago awaiting approval, but given everything currently going on with Twitter as a company this took much longer and I had to rerequest it as the keys become invalid. While I was able to get a GPT-3 API access key, it is limited to $18 credit. OpenAI  will not let you generate new content very quickly, so if you are using the generate functions in the notebook, please be mindful it may take some time and/or disconnect from openai. I know it is not best practice to leave your API key visible, however, it is currently difficult to obtain an openai key given the popularity of ChatGPT, several of my peers were unsuccessful so I thought it best to leave it so you can try it.

Another challenge was following the various Azure infrastructure gallery tutorials  https://github.com/Azure/azure-ai-personalized-offers/tree/master/Manual%20Deployment%20Guide I looked at and went through several tutorials familiarizing myself with the various components. However this requires knowledge of C# and a lot of editing of JSON files to reconfigure this to the current SDK version. I am not sure why these are still listed as up-to-date resources, but definitely a growing and learning pain. 

Please go to docker to access the image and create your own container. This will create both the churn model and uses GPT-3 (unfortunately microsoft's own GPT-3 licensing is not made public to all. Again another request I submitted but was denied for. Hence I had to find a work around which is detailed in the container. This actually generates the IT inspired educational content for our marketing promotional emails. The emails are created in bulk, with the intent that the legal team can quickly review multiple emails at once and there is less back and forth between them and the campaign team.

docker pull srajen1/705.603_sneharajen:systemdesign


Resources:
https://www.theaidream.com/post/openai-gpt-3-understanding-the-architecture#:~:text=OpenAI%20GPT%2D3%20Architecture&text=The%20largest%20version%20GPT%2D3,that%20it%20is%20quite%20larger.
https://dev.to/born2learn/deploy-your-custom-ai-models-on-azure-machine-learning-service-2bd6
https://accessibleai.dev/post/registering-pandas-dataframes-as-azure-datasets/
https://learn.microsoft.com/en-us/azure/machine-learning/v1/tutorial-power-bi-custom-model#prerequisites
https://www.youtube.com/watch?v=K3x0VT4ncaE
https://www.youtube.com/watch?v=aNI8pMjzgqg
https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml
https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction
https://www.analyticsvidhya.com/blog/2021/10/customer-churn-prediction-using-artificial-neural-network/
