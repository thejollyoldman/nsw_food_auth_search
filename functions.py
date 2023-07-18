# define function for parsing to clean up the below a bit
def my_parser(search_str, start_str, end_str):
	return(search_str[search_str.find(start_str) + len(start_str) : search_str.find(end_str, search_str.find(start_str) + len(start_str))] or "")

# define function that ranks and pulls the most relevant context out
import pandas as pd, numpy as np, numpy.linalg as npl, openai

def strings_ranked_by_relatedness(query: str, df: pd.DataFrame, relatedness_fn=lambda a, b: np.dot(a, b)/(npl.norm(a)*npl.norm(b)), top_n: int = 2, embedding_model = "text-embedding-ada-002",  threshold = 0.3) -> tuple[list[str], list[float]]:    

	
	"""Returns a list of strings and relatednesses, sorted from most related to least."""
	# pass the query to openAI embedding model to create an embedding
	query_embedding_response = openai.Embedding.create(
		model=embedding_model,
		input=query,
	)
	
	# extract embedding
	query_embedding = query_embedding_response["data"][0]["embedding"]
	
	# iterate over rows of the df (search_data), and pull the relevant text and the cosine similarity 
	strings_and_relatednesses = [
		(row["text"], relatedness_fn(query_embedding, row["embedding"]))
		for i, row in df.iterrows()
	]

	filtered_strings_and_relatednesses = [
    (text, embedding) for text, embedding in strings_and_relatednesses if embedding > threshold
	]

	# put conditional for a minimal threshold for similarity	
	if len(filtered_strings_and_relatednesses) == 0:
		return "", -1
		
	else:
		filtered_strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
		
		strings, relatednesses = zip(*filtered_strings_and_relatednesses)

		return strings[0:top_n], relatednesses[0:top_n]

# load OpenAI llm with 0.5 temperature
from langchain import PromptTemplate
from langchain import OpenAI

def load_LLM():
	llm = OpenAI(model_name="gpt-3.5-turbo", temperature = .1)
	return llm

# gets text from the streamlit app
import streamlit as st	

def get_text():
	# input_text = st.text_area(label="", placeholder="For e.g. 'ElJannah Punchbowl'", key="question")
	# return input_text
	
	# Using the "with" syntax
	with st.form(key='my_form'):
		input_text = st.text_input(label="", placeholder="For e.g. 'ElJannah Punchbowl'", key="question")
		submit_button = st.form_submit_button(label='Go')
		return input_text
