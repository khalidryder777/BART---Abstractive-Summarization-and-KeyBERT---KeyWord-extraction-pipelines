# Pipelines
# 1. BART---Abstractive-Summarization
# 2. KeyBERT---KeyWord-extraction


### 3. After this, both have common steps of doing [STS - Semantic Text Similarity]([https://pages.github.com/](https://github.com/khalidryder777/Semantic-Similarity---Higging-Face-pretrained-Transformers)) between the original and the output.

## 1. BART---Abstractive-Summarization
  #### a. Two methods to summarize text: 
   **Extractive summarization** involves selecting the most important sentences or phrases from the original text and concatenating them to form a summary. 
   This approach is considered to be a more straightforward and computationally simpler method, as it does not require an understanding of the underlying 
   meaning of the text.

**Abstractive summarization**, on the other hand, involves generating a new and condensed version of the original text that captures its essence. This approach requires a deeper understanding of the text and often involves generating new words, phrases, or sentences that are not present in the original text.

#### b. Why abstractive Summarization?
Abstractive summarization can be a better choice in scenarios where a more expressive summary is needed, and where a deeper understanding of the text is required. We'll be using a pretrained BART model provided on the Hugging Face website. 

#### c. About the model: "philschmid/bart-large-cnn-samsum"
This model is a pre-trained abstractive summarization model based on the BART (Bayesian Attention-based Recurrent Transformer) architecture. BART is based on the Transformer architecture and uses attention mechanisms to weight the importance of different parts of the input text when generating the summary. This specific model is fine-tuned on the "CNN/DailyMail" summarization dataset by Phil Schmid.

#### d. Pipeline Explained


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










