# Pipelines
# 1. BART---Abstractive-Summarization
# 2. KeyBERT---KeyWord-extraction


### 3. After this, both have common steps of doing [STS - Semantic Text Similarity](https://github.com/khalidryder777/Semantic-Similarity---Higging-Face-pretrained-Transformers) between the original and the output.

## 1. BART---Abstractive-Summarization
  ### a. Two methods to summarize text: 
   **Extractive summarization** involves selecting the most important sentences or phrases from the original text and concatenating them to form a summary. 
   This approach is considered to be a more straightforward and computationally simpler method, as it does not require an understanding of the underlying 
   meaning of the text.

**Abstractive summarization**, on the other hand, involves generating a new and condensed version of the original text that captures its essence. This approach requires a deeper understanding of the text and often involves generating new words, phrases, or sentences that are not present in the original text.

### b. Why abstractive Summarization?
Abstractive summarization can be a better choice in scenarios where a more expressive summary is needed, and where a deeper understanding of the text is required. We'll be using a pretrained BART model provided on the Hugging Face website. 

### c. About the model: "philschmid/bart-large-cnn-samsum"
This model is a pre-trained abstractive summarization model based on the BART (Bayesian Attention-based Recurrent Transformer) architecture. BART is based on the Transformer architecture and uses attention mechanisms to weight the importance of different parts of the input text when generating the summary. This specific model is fine-tuned on the "CNN/DailyMail" summarization dataset by Phil Schmid.

### d. Pipeline Explained
## i. Install and Import required libraries and modules:
We used Jupyter Notebook from Anaconda distribution here. Please type the following in a notebook in Jupyter Notebook to install the libraries. Alternatively, the libraries can also be installed using the Conda terminal.
```python
!pip install transformers
from transformers import pipeline
import pandas as pd
import numpy as np
```
## ii. Initialize the summarization pipeline with the pre-trained BART model: 'philschmid/bart-large-cnn-samsum' 
```python
#Hyperparameters: min_length = 5, max_length = 30
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", min_length = 5, max_length = 30)
```

## iii. Testing some inputs
```python
conversation1 = '''I imagine, that I was friend with Ariel.
I imagine we were going to get food.
It can also be that we were walking to class together or commuting together.'''

conversation2 = '''I picture the journey to Sam's house on a sunny day in New York. 
I picture all of the different people I pass in their own worlds, and the cute apartments and shops I look into as I walk by. 
I imagine myself making it to his house and it being warm out from the sun so we hang out in the backyard of his brownstone on a hammock.'''

conversation3 = '''I met with Ariel and we decided that we were going to go on a walk to explore the city. 
Ariel and I first decided that we were going to walk and get pizza before we went to sit in the park together. 
After getting pizza, we walked toward the park to enjoy the beautiful weather outside. 
Upon getting to the park, we sat and talked for hours, catching up.'''

x = summarizer(conversation1)
y = summarizer(conversation2)
z = summarizer(conversation3)

print(x)
print(y)
print(z)
```
    Output:
    [{'summary_text': 'I imagine that Ariel and I were going to get food together.'}]
    [{'summary_text': "Sam and I will hang out in the backyard of Sam's brownstone on a hammock when it's warm out."}]
    [{'summary_text': 'Ariel and I went for a walk and got pizza before sitting in the park together.'}]

## iV. Load the dataset
```python
df = pd.read_csv('/content/sample_data/ego_text.csv', encoding='cp1252')

#Check out some of it
df.head()
```

## v. Using summarizer to generate summaries of the input dataframe columns
```python
# We have 4 such columns in our dataframe, so repeat for each column(no. 1,3,5,7) by changing the index values in '.iloc' 
x1=[]
for i in range(len(df)):
  sentences = []
  sentences.append(df.iloc[:,1][i])
  x = summarizer(sentences)
  x1.append(x)
  
y1=[]
for i in range(len(df)):
  sentences = []
  sentences.append(df.iloc[:,3][i])
  y = summarizer(sentences)
  y1.append(y)
  
z1=[]
for i in range(len(df)):
  sentences = []
  sentences.append(df.iloc[:,5][i])
  z = summarizer(sentences)
  z1.append(z)
  
a1=[]
for i in range(len(df)):
  sentences = []
  sentences.append(df.iloc[:,7][i])
  a = summarizer(sentences)
  a1.append(a)
 
print(x1)
print(y1)
print(z1)
print(a1)
```

      Output samples:
      [[{'summary_text': 'I imagine Ariel and I were going to get food together.'}], [{'summary_text': 'Sam and I are eating pancakes at the dinner table.'}]
      [[{'summary_text': 'I visited Sam and Jessie between classes at their dorm and worked on their assignments. We went to eat before class and listened to'}]
      [[{'summary_text': 'Taylor ate with Ariel during their break between classes. They went to a local pizza shop to get a quick bite. They ate there'}]
      [[{'summary_text': 'Taylor talked with Sam and Jessie as they waited for their professor to start the lecture. They made plans about what they would do'}]

## Vi. Unpacking and storing the values iteratively
```python
x2 = []
for i in range(len(x1)):
    x2.append(x1[i][0]['summary_text'])
    
y2 = []
for i in range(len(y1)):
    y2.append(y1[i][0]['summary_text'])    

z2 = []
for i in range(len(z1)):
    z2.append(z1[i][0]['summary_text'])
    
a2 = []
for i in range(len(a1)):
    a2.append(a1[i][0]['summary_text'])
    
print(x2)
print(y2)
print(z2)
print(a2)
```
      Output samples:
      ['I imagine Ariel and I were going to get food together.',......]
      ['I visited Sam and Jessie between classes at their dorm and worked on their assignments. We went to eat before class and listened to', ....]
      ['Taylor ate with Ariel during their break between classes. They went to a local pizza shop to get a quick bite. They ate there',....]
      ['Taylor talked with Sam and Jessie as they waited for their professor to start the lecture. They made plans about what they would do',......]


## 2. KeyBERT---KeyWord-extraction
#### a. Other methods 
There are several method for keywords extraction such as TF-IDF and TextRank. These methods are rule-based or statistical in nature and do not use deep learning to understand the context and relationships between words in a text.

#### b. Why KeyBERT
KeyBERT is a deep learning-based method that leverages the power of pre-trained language models such as BERT to extract keywords from a text. 
It can be a better choice as it leverages deep learning to understand/infer the context and the complex relationships between words and ideas in the text.

#### c. About KeyBERT.
KeyBERT is a keyword extraction model based on the BERT (Bidirectional Encoder Representations from Transformers) architecture.
One of the key strengths of BERT architecture is its ability to process contextual information by looking at words both to the left and right of a target word which can help it in identifying the most important words or phrases in a text that convey its overall meaning.

#### d. Pipeline explained










