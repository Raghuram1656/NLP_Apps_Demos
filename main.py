import streamlit as st
import requests
import os, re
import string
import warnings
import validators
import time

#scrape
from newspaper import Article
from sklearn.feature_extraction import _stop_words


#Summarizer related pre-imports
import sumy
import transformers
import torch
from summarizer import Summarizer,TransformerSummarizer
from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer



warnings.filterwarnings("ignore")


"""
    (https://www.linkedin.com/in/siva-raghuram/)
"""

st.title ("Text Summarizer")
# st.code("x=2022")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')

# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)
# https://www.datacamp.com/tutorial/streamlit
#
#
# summarizers_dict = {"XLNet":1,"Transformers-Pipeline":2,"Bert":3,"LexRank":4,"Luhn":5,"Lsa":6,"KL-Sum":7}
# user_selection_summarizer=st.sidebar.selectbox('Pls Choose your Summarizer',tuple(summarizers_dict.keys()))
# st.write('you have selected',user_selection_summarizer)

summarizer_options = ["Bert","XLNet","LexRank","Luhn","Lsa","KL-Sum","BART"]
summarizer_type = st.sidebar.selectbox("Summarizer Options", options=summarizer_options, key='sbox')

def bert_values():
    global bert_ratio
    bert_ratio = st.sidebar.slider("Select Ratio For Bert",min_value=0.3,max_value=1.0)
if summarizer_type == 'Bert':
    bert_values()
          
    

st.markdown("""   
    In order to use the app:
    - Select the Preferred Text Summarization Model.
    - Paste the URL with your corpus.
    - Paste the long text into the textbox--(Under Implementation).
    - Upload a .txt or .docx or .pdf document through browsing option --(Under Implementation).
    - Text-Summarization away!! """
)

st.markdown("---")


### Functions Area 
#1. Get URL Text
def extract_text_from_url(url: str):
    
    '''Extract text from url'''
    
    article = Article(url)
    article.download()
    article.parse()
    
    # get text
    text = article.text
    ip_text = text
    
    # get article title
    title = article.title
    
    return title, text

def clear_text():
    st.session_state["text_url"] = ""
    st.session_state["text_input"]= ""


def clear_search_text():
    st.session_state["text_input"]= ""

@st.cache(suppress_st_warning=True)
def text_summarize(summarizer_type,text_to_be_summarized):
    ip_text = text_to_be_summarized
    global summarized_text
    match summarizer_type:
        case "Bert":
            # st.info('Hi, Please select Ration in sidebar, It is mandatory for Bert Model. Select and Relax! We will show output in a moment........')
            # select_ratio = st.sidebar.slider("Select Ratio For Bert",min_value=0.3,max_value=1.0)
            st.write(f'bert_ratio this model is taking is {bert_ratio}')
            bert_summarizer = Summarizer()
            summarized_text = bert_summarizer(ip_text,ratio=bert_ratio)
        case "XLNet":
            Xlnet_model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
            summarized_text = ''.join(Xlnet_model(ip_text,min_length=56,max_length=100))
        case "Transformers-Pipeline":
            pipeline_summarizer = pipeline("summarization",model="facebook/bart-large-xsum")
            summarized_text = pipeline_summarizer(ip_text,min_length=5,max_length=20)
        case "LexRank":
            lexrank_parser = PlaintextParser.from_string(ip_text,Tokenizer('english'))
            lexrank_summarizer = LexRankSummarizer()
            lexrank_summary = lexrank_summarizer(lexrank_parser.document,sentences_count=3)
            summarized_text = ''
            for sentence in lexrank_summary:
                summarized_text = summarized_text + str(sentence)
        case "Luhn":
            #Initialize the parser
            luhn_parser = PlaintextParser.from_string(ip_text,Tokenizer('english'))
            luhn_summarizer = LuhnSummarizer()
            luhn_summary = luhn_summarizer(luhn_parser.document,sentences_count=3)
            summarized_text = ''
            #print the summary
            for sentence in luhn_summary:
                summarized_text = summarized_text + str(sentence)
            st.snow()    
        case "Lsa":
            lsa_parser = PlaintextParser.from_string(ip_text,Tokenizer('english'))
            lsa_summarizer = LsaSummarizer()
            lsa_summary = lsa_summarizer(lsa_parser.document,sentences_count=3)
            summarized_text = ''
            #print the summary
            for sentence in lsa_summary:
                summarized_text = summarized_text + str(sentence)
        case "KL-Sum":
            kl_parser = PlaintextParser.from_string(ip_text,Tokenizer('english'))  
            Kl_summarizer = KLSummarizer()
            Kl_summary = Kl_summarizer(kl_parser.document,sentences_count=3)
            summarized_text = ''
            #print the summary
            for sentence in Kl_summary:
                summarized_text = summarized_text + str(sentence)        
        case "BART":
            tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            inputs = tokenizer.batch_encode_plus([ip_text],return_tensors='pt',max_length=512)
            summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
            bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summarized_text = bart_summary
        case _:
            st.write("Please select valid Summarizer")

    return summarized_text

#Workflow Area 

url_text = st.text_input("Please Enter a url here",value="https://en.wikipedia.org/wiki/Wiki",key='text_url',on_change=clear_search_text)

# st.markdown(
#     "<h3 style='text-align: center; color: red;'>OR</h3>",
#     unsafe_allow_html=True,
# )


if validators.url(url_text):
    #if input is URL
    title, text = extract_text_from_url(url_text)
    text_to_be_summarized = text
    # st.write(text)
    # st.write(text)

col1, col2 = st.columns(2)

with col1:
  summarize = st.button("Summarize",key='summarize_but', help='Click to Summarize!!')
  
with col2:
  clear = st.button("Clear Text Input", on_click=clear_text,key='clear',help='Click to clear the URL input and text')

if summarize:
    if summarizer_type:
        with st.spinner(text = f"Hi, We are loading the selected summarizer --- '{summarizer_type} 'into memory. This might take a few seconds"):
            #Animi
            # progress_bar = st.progress(0)
            # for i in range(100):
            #     time.sleep(0.01)
            #     progress_bar.progress(i+1)
            #response
            response_text = text_summarize(summarizer_type,text_to_be_summarized)
            #Animi
            if len(response_text) > 1:
               st.success("!Done")
               st.write(response_text)
            # st.balloons()

