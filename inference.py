#######################################################################################
# auth: ahmad sultani
# date: 23 june 2023
# desc: This script takes the prior embedding df called search_data and for a given
# prompt then:
# - turns the prompt into an embedding using the text-embedding-ada-002 api
# - does a cosine similarity of the prompt vector against the vector embedding database
# - Returns the text for the top 2 most relevant cases and cuts the size of the context to a maximum of 600 tokens
# - Places this as context in between the system prompt and the user question.
# - Passes this prompt into gpt 3.5 turbo to generate an answer
# - this is now a streamlit app
#######################################################################################

#######################################################################################
# import packages and functions, load embedding vectors
#######################################################################################

import os, pandas as pd, streamlit as st, langchain
from langchain import PromptTemplate
from langchain import OpenAI
from functions import *

search_data = pd.read_pickle('data/search_data.pkl')

# openAI key - do not save or commit file with key in it
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]

#######################################################################################
# langchain templae fed into prompt template with variables that can be adjusted (context
# + question)
# load OpenAI llm for use in streamlit app
#######################################################################################

template = """You are a helpful safe restuarant identification assistant providing a RESPONSE in Australian English. You will be given PRE-CONTEXT from the NSW Food Authority penalty list which contains the restaurants name, the address, the penalty amount, the penalty frequency and the combined offence descriptions. 

An example RESPONSE that is considered safe for a fictional restaurant called Threads is:

\"SAFE TO CONSUME FOOD FROM THREADS (ADDRESS: 200 HAPPY ROAD MOUNT LEWIS)
- The restaurant has only one penalty totalling $180 relating to fixtures and fittings being designed poorly for cleaning. 
- Given no penalties directly relating to food cleanliness and consumption, it is likely safe to eat here.\"

An example RESPONSE that is considered safe, but borderline for a restaurant called Akar Halal Meats:
\"SAFE TO CONSUME FOOD FROM AKAR HALAL MEATS (ADDRESS: 3 BEATRICE STREET AUBURN)

- The restaurant has two penalties totalling $1760.0 relating to the sale of food found to contain illegal preservative (sulphur dioxide) in their fat free beef mince and premium beef mince.
- Given the penalties are related to the use of illegal preservatives on a small sample of food, and not directly to food cleanliness and consumption, it is likely safe to eat here./"

An example RESPONSE that is considered unsafe for a fictional restaurant called Loops is:

\"UNSAFE TO CONSUME FOOD FROM LOOPS (ADDRESS: 100 SAD ROAD CAMDEN)
- The restaurant has over 6 penalties totalling $2000, all relating to hygiene and food preparation with multiple repeat offences. 
- Given all penalties relate to hygiene and food preperation, and the volume of penalties, you should exhibit caution if consuming food from this restaurant.
\"

ITEMS TO NOTE FOR RESPONSE:
- If the restaurant name is not similar to that provided by the USER QUESTION or if PRE-CONTEXT is empty, RESPONSE should be: \"The restaurant did not have penalties in the last year.\" 
- If you don't know the answer to the USER QUESTION from PRE-CONTEXT provided:, say \"I am unable to ascertain the answer from my data.\"

PRE-CONTEXT: \" {context} \"

USER QUESTION: "\ {user_query} \"

RESPONSE:
"""

prompt = PromptTemplate(
	input_variables = ["context", "user_query"],
	template = template
)

llm = load_LLM()

#######################################################################################
# Streamlit components with embedding search and prompt output component in app
#######################################################################################

#streamlit title and page config
st.set_page_config(page_title = "Food Authority Context Search", page_icon=":robot:")
st.title("Food Authority Context Search and Answer Sydney June 2023")

# create two columns with information for users
col1, col2 = st.columns(2)

with col1:
	
	st.markdown(
"""OpenAI embedding search and explanation for every NSW restaurant that is on the name and shame list over last 2 years as at June 2023, with information on why it flagged, and how severe and relevant to safe food consumption the finding actually is (i.e. if it is just ceiling lights are hard to clean, then more likely than not, nothing wrong with the food). 
 
Data scraped from [NSW Food Authority Penalty Notices](https://www.foodauthority.nsw.gov.au/offences/penalty-notices)"""
			   )
	
with col2: 
	st.image(image="data/Food_Authority.png", width=400)

# text box section
st.markdown("### Enter restaurant name and suburb")

question_input = get_text()

st.button(label = "Go", on_click=run_question(question_input))

# answer section
def run_question(question_input):
	# assuming there is a question input by user
	if question_input:
		
		# pull out top related embedding string (only top 1 for now). Top strings has been defined in functions.py
		top_strings = strings_ranked_by_relatedness(query = question_input, df = search_data, top_n = 1, threshold = 0.3)
	
		# craft final prompt with context, top_strings[0] retrieves first column of text only, not embedding value
		prompt_with_context = prompt.format(context = top_strings[0], user_query = question_input)
		
		# pass prompt into openAI llm
		answer = llm(prompt_with_context)
	
		# write the context that is found
		#st.write(top_strings[0])

		st.markdown("### Safety answer")
		
		st.write(answer)
	
	
