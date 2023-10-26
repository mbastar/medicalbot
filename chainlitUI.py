import os
import pandas as pd
from datasets import load_dataset
import openai
import pinecone
from tqdm.auto import tqdm
from nemoguardrails import LLMRails, RailsConfig
import chainlit as cl
from dotenv import load_dotenv

# Initialize environment variables
def initialize_resources():
    load_dotenv()

    # Load data
    data = load_dataset("MattBastar/Medicine_Details", split="train")

    # Your original code for generating the uid
    data = data.map(lambda x: {'uid': f"{replace_non_ascii(x['Composition'])}-{replace_non_ascii(x['Medicine Name'])}"})

    data = data.to_pandas()
    data = data[['uid', 'Medicine Name', 'Composition', 'Uses', 'Side_effects']]

    # Initialize Pinecone
    embed_model_id = "text-embedding-ada-002"
    api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
    env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"
    pinecone.init(api_key=api_key, environment=env)
    index_name = "nemo-guardrails-rag-with-actions"
    index = pinecone.Index(index_name)

    # Embedding and Upsert
    batch_size = 200

    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data[i:i_end]
        ids_batch = batch['uid'].tolist()
        texts = batch['Composition'].tolist()
        res = openai.Embedding.create(input=texts, engine=embed_model_id)
        embeds = [record['embedding'] for record in res['data']]
        metadata = [{
            'Medicine Name': row['Medicine Name'],
            'Composition': row['Composition'],
            'Uses': row['Uses'],
            'Side_effects': row['Side_effects'],
        } for index, row in batch.iterrows()]
        to_upsert = list(zip(ids_batch, embeds, metadata))
        index.upsert(vectors=to_upsert)
        
    return embed_model_id, index

       
# Function to replace common non-ASCII characters with their closest ASCII equivalents 
def replace_non_ascii(text):
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N',
            'ü': 'u', 'Ü': 'U',
            'ß': 'ss',
            'ø': 'o', 'Ø': 'O',
            'æ': 'ae', 'Æ': 'AE',
            'œ': 'oe', 'Œ': 'OE',
            '’': "'",
            '–': '-',
            '’': "'",
        }
        return ''.join(replacements.get(char, char) for char in text)

async def retrieve(query: str, embed_model_id: str, index) -> list:
    # create query embedding
    res = openai.Embedding.create(input=[query], engine=embed_model_id)
    xq = res['data'][0]['embedding']
    # get relevant contexts from pinecone
    res = index.query(xq, top_k=5, include_metadata=True)
    # get list of retrieved texts
    contexts = [x['metadata']['Composition'] for x in res['matches']]
    return contexts

async def rag(query: str, contexts: list, embed_model_id: str) -> str:
    print("> RAG Called")  # we'll add this so we can see when this is being used
    context_str = "\n".join(contexts)
    # place query and contexts into RAG prompt
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context_str}

    Query: {query}

    Answer: """
    # generate answer
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100
    )
    return res['choices'][0]['text']

# Initialize resources
embed_model_id, index = initialize_resources()

#initialize configs for guardrails

yaml_content = """
models:
- type: main
  engine: openai
  model: text-davinci-003
"""

rag_colang_content = """
# define limits
# Define invalid questions
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm an AI assistant focused on medicine, I cannot answer political questions."
    "My role is to answer medicine-related questions using trusted data sources."  

define flow politics
    user ask politics  
    bot answer politics
    bot offer help

# Define medicine info questions 
define user ask medicine
    "what are the side effects of ibuprofen?"
    "is there a substitute for penicillin?"
    "what class of medicine is metformin?"
    "how is aspirin used?"

# Define medicine Q&A flow
define flow medicine
    user ask medicine
    $contexts = execute retrieve(query=$last_user_message)  
    $answer = execute rag(query=$last_user_message, contexts=$contexts)
    bot $answer
"""

# initialize rails config
config = RailsConfig.from_content(
    colang_content=rag_colang_content,
    yaml_content=yaml_content
)
# create rails
rag_rails = LLMRails(config)

rag_rails.register_action(action=retrieve, name="retrieve")
rag_rails.register_action(action=rag, name="rag")

no_rag_colang_content = """
# define limits
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm a shopping assistant, I don't like to talk of politics."
    "Sorry I can't talk about politics!"

define flow politics
    user ask politics
    bot answer politics
    bot offer help
"""

# initialize rails config
config = RailsConfig.from_content(
    colang_content=no_rag_colang_content,
    yaml_content=yaml_content
)
# create rails
no_rag_rails = LLMRails(config)

@cl.on_message
async def main(message: cl.Message):
    try:
        user_query = message.content
        contexts = await retrieve(user_query, embed_model_id, index)
        answer = await rag(user_query, contexts, embed_model_id)
        await cl.Message(content=f"Answer: {answer}").send()
    except Exception as e:
        print(f"Error in main: {e}")
        await cl.Message(content="An error occurred. Please try again.").send()
    
    