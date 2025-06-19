# Multimodal_Rag



Multimodal RAG = Retrieval-Augmented Generation that works with more than just text, like images, PDFs, audio, or videos.

Think of it like this:
A smart assistant that can search through not just documents, but also images or videos, and then generate a helpful response combining all that info.
RAG normally deals with text retrieval + text generation
Multimodal RAG can handle text + other media (like image-to-text, audio-to-text, etc.)

To perform multimodal rag, you can find several approaches on the internet or blogs.
But I find it quite easy by going through 3rd approach, which is Combine Summaries + Raw Images.



Install Dependencies:

Here is requirement,txt file:

#Install these required dependencies to run this notebook
# ! pip install "unstructured[all-docs]" pillow pydantic lxml pillow matplotlib
# !sudo apt-get update
# !sudo apt-get install poppler-utils
# !sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
# !pip install unstructured-pytesseract
# !pip install tesseract-ocr
# !pip install python-dotenv==1.0.0
# !pip install requests
# !pip install sseclient-py==1.8.0
# !pip install pdf2image==1.17.0
# !pip install langchain-sambanova
# !pip install openai




STEP1.Extract the images and text and tables.

Use the unstructured library to extract structured data (text, tables, images) from an unstructured PDF document.

The partition_pdf function is used to extract the data, and the extracted elements are stored in the raw_pdf_elements variable.

Code snippet:

from unstructured.partition.pdf import partition_pdf

raw_pdf_elements=partition_pdf(
    filename="/Users/Desktop/ML_PRACTICE/data/NIPS-2017-attention-is-all-you-need-Paper.pdf",     #put pdf path 
    strategy="hi_res",
    extract_images_in_pdf=True,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=False,
    extract_image_block_output_dir="image_store",   #store the images and tables in folder
  )



STEP2.Stored extracted the images and text and tables.

Loops through the extracted elements and stores them in separate lists based on their type (e.g., Header, Footer, Title, NarrativeText, Text, ListItem, Image, Table).

The extracted images, text, and tables are stored in separate lists (e.g., img, table, NarrativeText).

Code snippet:

Header=[]
Footer=[]
Title=[]
NarrativeText=[]
Text=[]
ListItem=[]


for element in raw_pdf_elements:
  if "unstructured.documents.elements.Header" in str(type(element)):
            Header.append(str(element))
  elif "unstructured.documents.elements.Footer" in str(type(element)):
            Footer.append(str(element))
  elif "unstructured.documents.elements.Title" in str(type(element)):
            Title.append(str(element))
  elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
            NarrativeText.append(str(element))
  elif "unstructured.documents.elements.Text" in str(type(element)):
            Text.append(str(element))
  elif "unstructured.documents.elements.ListItem" in str(type(element)):
            ListItem.append(str(element))




Save extracted Images.Text and tables in the of list 

A] Extracted Image:

img=[]
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Image" in str(type(element)):
        img.append(str(element))

        
#optional for debugging purpose 
len(img)
for i in range(len(img)):
    print(i, img[i])


B]Extracted Table:

table=[]
for element in raw_pdf_elements:
  if "unstructured.documents.elements.Table" in str(type(element)):
            table.append(str(element))

#optional for debugging purpose 
len(table)
for i in range(len(table)):
    print(i, table[i])


C]B]Extracted Text:

NarrativeText=[]
for element in raw_pdf_elements:
    if "unstructured.documents.elements.NarrativeText" in str(type(element)):
            NarrativeText.append(str(element))

#optional for debugging purpose           
len(NarrativeText)
for i in range(len(NarrativeText)):
    print(i, NarrativeText[i])



STEP3.Summary of image,text,and table

Uses a Sambanova multimodal Large Language Model (LLM) to turn images and text data from a document into descriptive text summaries.

The model_wrapper.py file is imported, which contains a class SambaNovaCloud that interacts with the Sambanova Cloud API.

The  SambaNovaCloud class to generate short descriptive summaries of the extracted images, text, and tables.


A]Text and Table Summary:

Chaining and Prompt Engineering

Chaining and prompt engineering are two important techniques that can be used to improve the performance of Multimodal RAG models. Chaining involves using the output of one model as the input to another model, while prompt engineering involves designing prompts that are optimized for a specific task or model.

To use chaining and prompt engineering with Sambanova APIs and models, you can use the ChatPromptTemplate class from the LangChain library. This class provides a simple and efficient way to define and format prompts for chat-based LLMs.


Code snippet:

Download this file 

File: 

import sys
sys.path.append('/Users/shivanim/Desktop/ML_PRACTICE/MULTIMODAL_RAG/model_wrapper.py')  # Add the folder path
import model_wrapper

from model_wrapper import SambaNovaCloud
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
sambanova_api_key = "SAMBANOVA_API_KEY"
os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

from langchain_sambanova import ChatSambaNovaCloud

llm = ChatSambaNovaCloud(
    model="Meta-Llama-3.3-70B-Instruct",
    max_tokens=2024,
    temperature=0.7,
    top_p=0.01,
)

Imports ChatPromptTemplate, a LangChain class that lets you define and format prompts for chat-based LLMsThese lines convert the plain text strings into LangChain-compatible templates.You can now use .format_messages(element="...") to inject content into {element}. 

#####Chaining
from langchain_core.prompts import ChatPromptTemplate


prompt_text="""You are a helpful assistant tasked with summarizing text.Give a concise summary of the NarrativeText.NarrativeText {element}"""

prompt_table="""You are a helpful assistant tasked with summarizing tables .Give a concise summary of the table.Table {element}"""

# These lines convert the plain text strings into LangChain-compatible templates.
# You can now use .format_messages(element="...") to inject content into {element}.
prompt_text = ChatPromptTemplate.from_template(prompt_text)
prompt_table=ChatPromptTemplate.from_template(prompt_table)



 Creating a data flow pipeline, where each component transforms the input and passes it to the next:

 1. {"element": lambda x: x}:The key "element" matches the {element} placeholder in your prompt templates. 

  2.| prompt_text or | prompt_table:This takes the formatted dictionary and applies it to the prompt using ChatPromptTemplate. 

 3. | llm:This sends the prompt to your LLM (Language Model), e.g., sambanova, etc., and gets a response. 

 4. | StrOutputParser():This parses the output and converts it into a plain string (instead of an LLM message object). 

text_summarize_chain = {"element": lambda x: x} | prompt_text | llm | StrOutputParser()
table_summarize_chain= {"element": lambda x: x} | prompt_table | llm | StrOutputParser()

Batch() method in LangChain to summarize multiple narrative texts in parallel or controlled batches.

{'max_concurrency': 1}:his sets the maximum number of parallel executions to 1, meaning it processes one input at a time (sequentially).

NarrativeText_summaries = []
if NarrativeText:
    NarrativeText_summaries = text_summarize_chain.batch(NarrativeText, {'max_concurrency': 1})  #This sets the maximum number of parallel executions to 1, meaning it processes one input at a time (sequentially).

table_summaries = []
table_summaries=table_summarize_chain.batch(table,{"max_concurrency": 3})



B]Image Summary:

Uses the SAMBANOVA API to summarize images. It takes a list of image files, encodes them as base64 strings, and then uses the image_summarizer function to generate summaries for each image.


code snippet:

import openai
import base64
import os 
from langchain_core.messages import HumanMessage
from IPython.display import HTML, display

client = openai.OpenAI(
    base_url="https://api.sambanova.ai/v1", 
    api_key="SAMBANOVA_API_KEY")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
    
def image_summarizer(prompt,image_base64):
    response = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    )

    return response.choices[0].message.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            generated_summary = image_summarizer(prompt,base64_image)
            print(generated_summary)
            image_summaries.append(image_summarizer(prompt,base64_image))

    return img_base64_list, image_summaries


# Image summaries
img_base64_list, image_summaries = generate_img_summaries("/Users/shivanim/Desktop/ML_PRACTICE/image_store") #image stored path



STEP4.Adding to the Vector Store

The 4 steps to create a MultiVectorRetriever involve initializing a storage tier, creating a vector store, creating a MultiVectorRetriever instance, and adding documents to the retriever, which enables fast and semantically accurate retrieval of text, tables, and images while preserving access to original content.



Code snippet:

import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
# from langchain_community.embeddings import SambaStudioEmbeddings
from langchain_sambanova import SambaNovaCloudEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

Create Embedding using SambaNovaCloudEmbeddings:

from langchain_sambanova import SambaNovaCloudEmbeddings

embeddings = SambaNovaCloudEmbeddings(
    model="E5-Mistral-7B-Instruct",sambanova_api_key="SAMBANOVA_API_KEY")

# This function creates a MultiVectorRetriever that indexes embeddings of summaries (text, table, and image) in a vector store, while storing the original content in an in-memory docstore. When queried, the retriever uses vector similarity on the summaries to retrieve the most relevant results and maps them back to the original full documents. This enables fast, semantically accurate retrieval while preserving access to detailed source data.



def create_multi_vector_retriever(vectorstore,text_summaries, texts, table_summaries, table, image_summaries,img
):
    """
    Create a retriever that indexes the summary but returns the original image or text.
    """


  # Initialize the storage tier
    store = InMemoryStore()
    id_key = "doc_id"

        # vectorstore = Chroma(
    #     # collection_name='summaries',embedding_function=embeddings )
    # print(vectorstore)
    # print(table_summaries)
    # print(texts)

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(vectorstore=vectorstore, 
                                        docstore=store, 
                                        id_key=id_key)
                                        # search_kwargs={'k': 2})  

        
    print(retriever)

    def add_documents(retriever, doc_summaries, doc_contents):
        print(doc_contents)
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        print(retriever)
        summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
        print(summary_docs)

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

        # Add texts, tables, and images
        # Check that text_summaries is not empty before adding
    if text_summaries:
        # print(text_summaries)
        add_documents(retriever, text_summaries, texts)
       
    # Check that table_summaries is not empty before adding
    if table_summaries:
        # print(table_summaries)
        add_documents(retriever, table_summaries, table)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, img)
      
    return retriever


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name='summaries',embedding_function=embeddings,persist_directory="sample-rag-multi-modal" )
print(vectorstore)

# def create_multi_vector_retriever(vectorstore,text_summaries, texts, table_summaries, table, image_summaries, img
# ):
# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    NarrativeText_summaries,
    NarrativeText,
    table_summaries,
    table,
    image_summaries,
    img_base64_list,
)

retriever_multi_vector_img



STEP 4:Preparing Input Data for Multimodal Retrieval: A Key to Accurate Results

In the field of multimodal retrieval, it's essential to preprocess input data to prepare it for the retrieval system. This involves splitting the input data into separate lists for images and text, resizing images to a consistent size, and formatting the input data into a prompt for the language model.

Preprocessing input data is crucial for multimodal retrieval because it allows the retrieval system to effectively retrieve relevant information and generate a response to the user's question. By splitting the input data into separate lists for images and text, resizing images to a consistent size, and formatting the input data into a prompt for the language model, we can ensure that the retrieval system has the necessary information to provide accurate and relevant results.





Code snippet:

import io
import re

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(400, 300))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are analyst and advice.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answer related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]



Step 5:Multimodal RAG Chain in Information Retrieval

The Multimodal RAG Chain is used after retrieval to generate a response to the user's query by combining the retrieved text and images. This chain of operations enables the system to format a multimodal prompt and pass it to the LLM for processing. The resulting response is then parsed into a plain string, providing a final answer to the user's question





Code snippet:

def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatSambaNovaCloud(model="Llama-4-Maverick-17B-128E-Instruct",temperature=0.7,top_p=0.01)
    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain


chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)



STEP 6:Querying the Multimodal RAG Chain

The multimodal RAG chain can generate different types of responses depending on the query and the data stored in the vector store. There are three main types of responses:





Code snippet:

Type1:Multimodal Response (Image and Text):

This type of response includes both text and images, retrieved from the vector store, in response to a user's query. The query_multimodal_rag function generates this type of response.

# ---Text and Image---

def query_multimodal_rag(query_text):
    response = chain_multimodal_rag.invoke(query_text)
    print(response)
    
    # Get docs separately (already retrieved)
    docs = retriever_multi_vector_img.invoke(query_text, limit= 6)
    
    # Display all images retrieved and detected as base64 images
    for img in split_image_text_types(docs)["images"]:
        plt_img_base64(img)

# User can enter the query
query_text = input("Enter your query: ")
query_multimodal_rag(query_text)



Type2:Image-Only Response:


This type of response only includes images, retrieved from the vector store, in response to a user's query. The code snippet that queries by image and displays the retrieved images using IPyImage generates this type of response.

# --- Query by Image ---
# The response provides a visual summary of the information related to the query.
import base64
from io import BytesIO
from IPython.display import display, Image as IPyImage
import re

chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
query_text = "What is mult-head-attention?"

docs = retriever_multi_vector_img.invoke(query_text, limit=6)

def is_base64_image(text):
    return isinstance(text, str) and re.match(r'^/9j|^iVB', text.strip())  # JPEG or PNG signatures

for i, doc in enumerate(docs):
    print(f"{i}:", end=" ")

    if is_base64_image(doc):
        try:
            # Display inline image
            image_data = base64.b64decode(doc)
            display(IPyImage(data=image_data))
        except Exception as e:
            print(f"[Error displaying image]: {e}")
    else:
        print(doc)
print(docs)



Type3:Text-Only Response:

This type of response only includes text, retrieved from the vector store, in response to a user's query. The code snippet that queries by text and prints the response using print(response_text) generates this type of response.

# --- Query by TEXT ---

query_text = "What is mult-head-attention?"

response_text = chain_multimodal_rag.invoke(query_text)
print("Response to text query:")
print(response_text)



For your reference here are the notebook and and sample pdf docs please refer to github

By following these steps, Multimodal RAG can provide a more accurate and relevant response to the user's query, and improve the overall user experience.



Thank you and happy learning……………..! 
