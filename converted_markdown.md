# Converted PDF to Markdown

What Is Multimodal RAG? Simple Guide Using Sambanova APIs and models.

Multimodal RAG = Retrieval-Augmented Generation that works with more than just text, like images, PDFs,

audio, or videos.

Think of it like this:

A smart assistant that can search through not just documents, but also images or videos, and then generate a helpful

response combining all that info.

RAG normally deals with text retrieval + text generation

Multimodal RAG can handle text + other media (like image-to-text, audio-to-text, etc.)

To perform multimodal rag, you can find several approaches on the internet or blogs.

But I find it quite easy by going through 3rd approach, which is Combine Summaries + Raw Images.

Install Dependencies:

Here is requirement,txt file:

requirem
19 Jun 2025, 11:27 am

ent.txt

1 #Install these required dependencies to run this notebook

2 # ! pip install "unstructured[all-docs]" pillow pydantic lxml pillow matplotlib

3 # !sudo apt-get update

4 # !sudo apt-get install poppler-utils

5 # !sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng

tesseract-ocr-script-latn

6 # !pip install unstructured-pytesseract

7 # !pip install tesseract-ocr

8 # !pip install python-dotenv==1.0.0

9 # !pip install requests

10 # !pip install sseclient-py==1.8.0

11 # !pip install pdf2image==1.17.0

12 # !pip install langchain-sambanova

13 # !pip install openai

14

STEP1.Extract the images and text and tables.

Use the unstructured library to extract structured data (text, tables, images) from an unstructured PDF document.

The  partition_pdf  function is used to extract the data, and the extracted elements are stored in the  raw_pdf_elements

variable.

Code snippet:

1 from unstructured.partition.pdf import partition_pdf

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

1 raw_pdf_elements=partition_pdf(

filename="/Users/Desktop/ML_PRACTICE/data/NIPS-2017-attention-is-all-you-need-Paper.pdf",     #put pdf path

strategy="hi_res",

extract_images_in_pdf=True,

extract_image_block_types=["Image", "Table"],

extract_image_block_to_payload=False,

extract_image_block_output_dir="image_store",   #store the images and tables in folder

2

3

4

5

6

7

8

)

STEP2.Stored extracted the images and text and tables.

Loops through the extracted elements and stores them in separate lists based on their type (e.g., Header, Footer, Title,

NarrativeText, Text, ListItem, Image, Table).

The extracted images, text, and tables are stored in separate lists (e.g.,  img ,  table ,  NarrativeText ).

Code snippet:

1 Header=[]

2 Footer=[]

3 Title=[]

4 NarrativeText=[]

5 Text=[]

6 ListItem=[]

7

8

9 for element in raw_pdf_elements:

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

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

Save extracted Images.Text and tables in the of list 

A] Extracted Image:

1 img=[]

2 for element in raw_pdf_elements:

if "unstructured.documents.elements.Image" in str(type(element)):

img.append(str(element))

3

4

5

6

7 #optional for debugging purpose 

8 len(img)

9 for i in range(len(img)):

print(i, img[i])

10

11

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

B]Extracted Table:

1 table=[]

2 for element in raw_pdf_elements:

if "unstructured.documents.elements.Table" in str(type(element)):

table.append(str(element))

3

4

5

6 #optional for debugging purpose 

7 len(table)

8 for i in range(len(table)):

print(i, table[i])

9

10

C]B]Extracted Text:

1 NarrativeText=[]

2 for element in raw_pdf_elements:

if "unstructured.documents.elements.NarrativeText" in str(type(element)):

NarrativeText.append(str(element))

3

4

5

6 #optional for debugging purpose           

7 len(NarrativeText)

8 for i in range(len(NarrativeText)):

9

print(i, NarrativeText[i])

STEP3.Summary of image,text,and table

Uses a Sambanova multimodal Large Language Model (LLM) to turn images and text data from a document into descriptive

text summaries.

The  model_wrapper.py  file is imported, which contains a class  SambaNovaCloud  that interacts with the Sambanova Cloud API.

The   SambaNovaCloud  class to generate short descriptive summaries of the extracted images, text, and tables.

A]Text and Table Summary:

Chaining and Prompt Engineering

Chaining and prompt engineering are two important techniques that can be used to improve the performance of

Multimodal RAG models. Chaining involves using the output of one model as the input to another model, while prompt

engineering involves designing prompts that are optimized for a specific task or model.

To use chaining and prompt engineering with Sambanova APIs and models, you can use the  ChatPromptTemplate  class

from the LangChain library. This class provides a simple and efficient way to define and format prompts for chat-based

LLMs.

Code snippet:

Download this file 

File: model_wrapper.py

1 import sys

2 sys.path.append('/Users/shivanim/Desktop/ML_PRACTICE/MULTIMODAL_RAG/model_wrapper.py')

# Add the folder path

3 import model_wrapper

1 from model_wrapper import SambaNovaCloud

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

2 from langchain_core.output_parsers import StrOutputParser

3 from langchain_core.prompts import ChatPromptTemplate

1 import os

2 sambanova_api_key = "SAMBANOVA_API_KEY"

3 os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

1 from langchain_sambanova import ChatSambaNovaCloud

2

3 llm = ChatSambaNovaCloud(

model="Meta-Llama-3.3-70B-Instruct",

max_tokens=2024,

temperature=0.7,

top_p=0.01,

4

5

6

7

8 )

Imports ChatPromptTemplate, a LangChain class that lets you define and format prompts for chat-based LLMsThese lines

convert the plain text strings into LangChain-compatible templates.You can now use .format_messages(element="...") to inject

content into {element}. 

1 #####Chaining

2 from langchain_core.prompts import ChatPromptTemplate

3

4

5 prompt_text="""You are a helpful assistant tasked with summarizing text.Give a concise summary of the

NarrativeText.NarrativeText {element}"""

6

7 prompt_table="""You are a helpful assistant tasked with summarizing tables .Give a concise summary of the

table.Table {element}"""

8

9 # These lines convert the plain text strings into LangChain-compatible templates.

10 # You can now use .format_messages(element="...") to inject content into {element}.

11 prompt_text = ChatPromptTemplate.from_template(prompt_text)

12 prompt_table=ChatPromptTemplate.from_template(prompt_table)

Creating a data flow pipeline, where each component transforms the input and passes it to the next:

1. {"element": lambda x: x}:The key "element" matches the {element} placeholder in your prompt templates. 

2.| prompt_text or | prompt_table:This takes the formatted dictionary and applies it to the prompt using ChatPromptTemplate. 

3. | llm:This sends the prompt to your LLM (Language Model), e.g., sambanova, etc., and gets a response. 

4. | StrOutputParser():This parses the output and converts it into a plain string (instead of an LLM message object). 

1 text_summarize_chain = {"element": lambda x: x} | prompt_text | llm | StrOutputParser()

2 table_summarize_chain= {"element": lambda x: x} | prompt_table | llm | StrOutputParser()

Batch() method in LangChain to summarize multiple narrative texts in parallel or controlled batches.

{'max_concurrency': 1}:his sets the maximum number of parallel executions to 1, meaning it processes one input at a time

(sequentially).

1 NarrativeText_summaries = []

2 if NarrativeText:

3

NarrativeText_summaries = text_summarize_chain.batch(NarrativeText, {'max_concurrency': 1})

#This sets the

maximum number of parallel executions to 1, meaning it processes one input at a time (sequentially).

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

1 table_summaries = []

2 table_summaries=table_summarize_chain.batch(table,{"max_concurrency": 3})

B]Image Summary:

Uses the SAMBANOVA API to summarize images. It takes a list of image files, encodes them as base64 strings, and then uses

the  image_summarizer  function to generate summaries for each image.

code snippet:

1 import openai

2 import base64

3 import os 

4 from langchain_core.messages import HumanMessage

5 from IPython.display import HTML, display

1 client = openai.OpenAI(

2

3

base_url="https://api.sambanova.ai/v1",

api_key="SAMBANOVA_API_KEY")

1 def encode_image(image_path):

with open(image_path, "rb") as image_file:

return base64.b64encode(image_file.read()).decode('utf-8')

2

3

4

5 def image_summarizer(prompt,image_base64):

response = client.chat.completions.create(

model="Llama-4-Maverick-17B-128E-Instruct",

messages=[

"role": "user",

"content": [

{"type": "text", "text": prompt},

{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}

]

{

}

]

)

return response.choices[0].message.content

6

7

8

9

10

11

12

13

14

15

16

17

18

19

1

2 def generate_img_summaries(path):

3

4

5

6

7

8

9

10

11

12

13

14

15

16

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

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

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32 # Image summaries

33 img_base64_list, image_summaries = generate_img_summaries("/Users/shivanim/Desktop/ML_PRACTICE/image_store")

#image stored path

STEP4.Adding to the Vector Store

The 4 steps to create a MultiVectorRetriever involve initializing a storage tier, creating a vector store, creating a

MultiVectorRetriever instance, and adding documents to the retriever, which enables fast and semantically accurate retrieval of

text, tables, and images while preserving access to original content.

Code snippet:

1 import uuid

2 from langchain.retrievers.multi_vector import MultiVectorRetriever

3 from langchain.storage import InMemoryStore

4 from langchain_chroma import Chroma

5 from langchain_core.documents import Document

6 # from langchain_community.embeddings import SambaStudioEmbeddings

7 from langchain_sambanova import SambaNovaCloudEmbeddings

8 from dotenv import load_dotenv

9 load_dotenv()

10 import os

Create Embedding using SambaNovaCloudEmbeddings:

1 from langchain_sambanova import SambaNovaCloudEmbeddings

2

3 embeddings = SambaNovaCloudEmbeddings(

4

model="E5-Mistral-7B-Instruct",sambanova_api_key="SAMBANOVA_API_KEY")

1 # This function creates a MultiVectorRetriever that indexes embeddings of summaries (text, table, and image)

in a vector store, while storing the original content in an in-memory docstore. When queried, the retriever

uses vector similarity on the summaries to retrieve the most relevant results and maps them back to the

original full documents. This enables fast, semantically accurate retrieval while preserving access to

detailed source data.

2

3

4

5 def create_multi_vector_retriever(vectorstore,text_summaries, texts, table_summaries, table,

image_summaries,img

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

6 ):

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

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

1 # The vectorstore to use to index the child chunks

2 vectorstore = Chroma(

3

collection_name='summaries',embedding_function=embeddings,persist_directory="sample-rag-multi-modal" )

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

4 print(vectorstore)

5

6 # def create_multi_vector_retriever(vectorstore,text_summaries, texts, table_summaries, table,

image_summaries, img

7 # ):

8 # Create retriever

9 retriever_multi_vector_img = create_multi_vector_retriever(

vectorstore,

NarrativeText_summaries,

NarrativeText,

table_summaries,

table,

image_summaries,

img_base64_list,

10

11

12

13

14

15

16

17 )

18

19 retriever_multi_vector_img

STEP 4:Preparing Input Data for Multimodal Retrieval: A Key to Accurate Results

In the field of multimodal retrieval, it's essential to preprocess input data to prepare it for the retrieval system. This involves

splitting the input data into separate lists for images and text, resizing images to a consistent size, and formatting the input data

into a prompt for the language model.

Preprocessing input data is crucial for multimodal retrieval because it allows the retrieval system to effectively retrieve

relevant information and generate a response to the user's question. By splitting the input data into separate lists for

images and text, resizing images to a consistent size, and formatting the input data into a prompt for the language model,

we can ensure that the retrieval system has the necessary information to provide accurate and relevant results.

Code snippet:

1 import io

2 import re

3

4 from IPython.display import HTML, display

5 from langchain_core.runnables import RunnableLambda, RunnablePassthrough

6 from PIL import Image

1 def plt_img_base64(img_base64):

"""Disply base64 encoded string as image"""

# Create an HTML img tag with the base64 string as the source

image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

# Display the image by rendering the HTML

display(HTML(image_html))

2

3

4

5

6

7

8

9 def looks_like_base64(sb):

"""Check if the string looks like base64"""

return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

10

11

12

13 def is_image_data(b64data):

14

15

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

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

header = base64.b64decode(b64data)[:8]

# Decode and get the first 8 bytes

for sig, format in image_signatures.items():

if header.startswith(sig):

return True

return False

except Exception:

return False

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32 def resize_base64_image(base64_string, size=(128, 128)):

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

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51 def split_image_text_types(docs):

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

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68 def img_prompt_func(data_dict):

69

70

71

72

73

"""

Join the context into a single string

"""

formatted_texts = "\n".join(data_dict["context"]["texts"])

messages = []

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

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

The Multimodal RAG Chain is used after retrieval to generate a response to the user's query by combining the retrieved text and

images. This chain of operations enables the system to format a multimodal prompt and pass it to the LLM for processing. The

resulting response is then parsed into a plain string, providing a final answer to the user's question

Code snippet:

1 def multi_modal_rag_chain(retriever):

"""

Multi-modal RAG chain

"""

# Multi-modal LLM

model = ChatSambaNovaCloud(model="Llama-4-Maverick-17B-128E-Instruct",temperature=0.7,top_p=0.01)

# RAG pipeline

chain = (

{

}

"context": retriever | RunnableLambda(split_image_text_types),

"question": RunnablePassthrough(),

| RunnableLambda(img_prompt_func)

| model

| StrOutputParser()

)

return chain

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

22 chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

STEP 6:Querying the Multimodal RAG Chain

The multimodal RAG chain can generate different types of responses depending on the query and the data stored in the vector

store. There are three main types of responses:

Code snippet:

Type1:Multimodal Response (Image and Text):

This type of response includes both text and images, retrieved from the vector store, in response to a user's query. The 

query_multimodal_rag  function generates this type of response.

1 # ---Text and Image---

2

3 def query_multimodal_rag(query_text):

response = chain_multimodal_rag.invoke(query_text)

print(response)

# Get docs separately (already retrieved)

docs = retriever_multi_vector_img.invoke(query_text, limit= 6)

# Display all images retrieved and detected as base64 images

for img in split_image_text_types(docs)["images"]:

plt_img_base64(img)

4

5

6

7

8

9

10

11

12

13

14 # User can enter the query

15 query_text = input("Enter your query: ")

16 query_multimodal_rag(query_text)

Type2:Image-Only Response:

This type of response only includes images, retrieved from the vector store, in response to a user's query. The code snippet that

queries by image and displays the retrieved images using IPyImage generates this type of response.

1 # --- Query by Image ---

2 # The response provides a visual summary of the information related to the query.

3 import base64

4 from io import BytesIO

5 from IPython.display import display, Image as IPyImage

6 import re

7

8 chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

9 query_text = "What is mult-head-attention?"

10

11 docs = retriever_multi_vector_img.invoke(query_text, limit=6)

12

13 def is_base64_image(text):

return isinstance(text, str) and re.match(r'^/9j|^iVB', text.strip())

# JPEG or PNG signatures

14

15

16 for i, doc in enumerate(docs):

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

17

18

19

20

21

22

23

24

25

26

27

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

28 print(docs)

Type3:Text-Only Response:

This type of response only includes text, retrieved from the vector store, in response to a user's query. The code snippet that

queries by text and prints the response using  print(response_text)  generates this type of response.

1 # --- Query by TEXT ---

2

3 query_text = "What is mult-head-attention?"

4

5 response_text = chain_multimodal_rag.invoke(query_text)

6 print("Response to text query:")

7 print(response_text)

For your reference here are the notebook and and sample pdf docs please refer to github

By following these steps, Multimodal RAG can provide a more accurate and relevant response to the user's query, and improve

the overall user experience.

Thank you and happy learning……………..!

Copyright © 2020-2023 SambaNova Systems, Inc. All rights reserved. Do not distribute without express written consent.

![Page 1](./images_md/page_1.png)

**OCR Text for Page 1:**
```
What Is Multimodal RAG? Simple Guide Using Sambanova APIs and models.

Multimodal RAG = Retrieval-Augmented Generation that works with more than just text, like images, PDFs,
audio, or videos. ?

@ Think of it like this:
A smart assistant that can search through not just documents, but also images or videos, and then generate a helpful
response combining all that info.
RAG normally deals with text retrieval + text generation

Multimodal RAG can handle text + other media (like image-to-text, audio-to-text, etc.)

To perform multimodal rag, you can find several approaches on the internet or blogs.

But I find it quite easy by going through 3rd approach, which is Combine Summaries + Raw Images.

Install Dependencies: ?

Here is requirement,txt file:

requirement.txt
19 Jun 2025, 11:27 am

1 #Install these required dependencies to run this notebook

2 # ! pip install "unstructured[all-docs]" pillow pydantic 1xml pillow matplotlib

3 # !sudo apt-get update

4 # !sudo apt-get install poppler-utils

5 # !sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng
tesseract-ocr-script-latn

6 # !pip install unstructured-pytesseract

7 # !pip install tesseract-ocr

8 # !pip install python-dotenv==1.0.0

9 # !pip install requests

10 # !pip install sseclient-py==1.8.0

11 # !pip install pdf2image==1.17.0

12 # !pip install langchain-sambanova

13 # !pip install openai

14

STEP1.Extract the images and text and tables. 2
e Use the unstructured library to extract structured data (text, tables, images) from an unstructured PDF document.

e The partition_pdf function is used to extract the data, and the extracted elements are stored in the raw_pdf_elements

variable.

Code snippet:

1 from unstructured.partition.pdf import partition_pdf
```

![Page 2](./images_md/page_2.png)

**OCR Text for Page 2:**
```
1 raw_pdf_elements=partition_pdf(

2 filename="/Users/Desktop/ML_PRACTICE/data/NIPS-2017-attention-is-all-you-need-Paper.pdf", #put pdf path
3 strategy="hi_res",

4 extract_images_in_pdf=True,

5 extract_image_block_types=["Image", "Table"],

6 extract_image_block_to_payload=False,

7 extract_image_block_output_dir="image_store", #store the images and tables in folder

8 )

STEP2.Stored extracted the images and text and tables. 2

e Loops through the extracted elements and stores them in separate lists based on their type (e.g., Header, Footer, Title,

NarrativeText, Text, ListItem, Image, Table).

« The extracted images, text, and tables are stored in separate lists (e.g., img, table, NarrativeText ).

Code snippet:

1 Header=[]

2 Footer=[]

3 Title=[]

4 NarrativeText=[]

5 Text=[]

6 ListItem=[]

7

8

9 for element in raw_pdf_elements:

10 if "unstructured.documents.elements.Header" in str(type(element)):
11) Header. append(stxr(element) )

12 elif "unstructured.documents.elements.Footer" in str(type(element)):
13 Footer. append(stxr(element) )

14 elif "unstructured.documents.elements.Title" in str(type(element)):
15) Title.append(stxr(element))

16, elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
17 NarrativeText .append(str(element) )

18 elif "unstructured.documents.elements.Text" in str(type(element)):
19 Text .append(str(element) )

20 elif "unstructured.documents.elements.ListItem" in stxr(type(element)):
21) ListItem.append(str(element) )

22

23)

24

Save extracted Images.Text and tables in the of list

A] Extracted Image:

img=[]
for element in raw_pdf_elements:
if "unstructured.documents.elements.Image" in str(type(element)):
img.append(stxr(element) )

#optional for debugging purpose
len(img)

for i in range(len(img)):

10 print(i, img[i])
```

![Page 3](./images_md/page_3.png)

**OCR Text for Page 3:**
```
BIE

al

OOO nn wD

xtracted Table:

table=[]
for element in raw_pdf_elements:
if "unstructured.documents.elements.Table" in str(type(element)):
table.append(str(element) )

#optional for debugging purpose

len(table)

for i in range(len(table)):
print(i, table[i])

C]BJExtracted Text:

Cod sO RWANYP

ST

NarrativeText=[]
for element in raw_pdf_elements:

if "unstructured.documents.elements.NarrativeText" in str(type(element)):

NarrativeText.append(str(element) )

#optional for debugging purpose

len(NarrativeText)

for i in range(len(NarrativeText)):
print(i, NarrativeText[i])

EP3.Summary of image,text,and table 7

« Uses a Sambanova multimodal Large Language Model (LLM) to turn images and text data from a document into descriptive

text summaries.

e The model_wrapper.py file is imported, which contains a class SambaNovaCloud that interacts with the Sambanova Cloud API.

e The SambaNovaCloud class to generate short descriptive summaries of the extracted images, text, and tables.

AlText and Table Summary:

Ci] Chaining and Prompt Engineering

Co

Chaining and prompt engineering are two important techniques that can be used to improve the performance of

Multimodal RAG models. Chaining involves using the output of one model as the input to another model, while prompt

engineering involves designing prompts that are optimized for a specific task or model.

To use chaining and prompt engineering with Sambanova APIs and models, you can use the ChatPromptTemplate class

from the LangChain library. This class provides a simple and efficient way to define and format prompts for chat-based

LLMs.

de snippet:

Download this file

File: &§ model_wrapper.py

1 import sys

2 | sys.path.append('/Users/shivanim/Desktop/ML_PRACTICE/MULTIMODAL_RAG/model_wrapper.py')

3 import model_wrapper

1 fxom model_wrapper import SambaNovaCloud

# Add the folder path
```

![Page 4](./images_md/page_4.png)

**OCR Text for Page 4:**
```
2 fxom langchain_core.output_parsers import StrOutputParser
3 from langchain_core.prompts import ChatPromptTemplate

1 import os
2. sambanova_api_key = "SAMBANOVA_API_KEY"
3 | os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

from langchain_sambanova import ChatSambaNovaCloud

11m = ChatSambaNovaCloud(
model="Meta-Llama-3.3-70B-Instruct",
max_tokens=2024,
temperature=0.7,
top_p=0.01,

Ordon nwne

Imports ChatPromptTemplate, a LangChain class that lets you define and format prompts for chat-based LLMsThese lines
convert the plain text strings into LangChain-compatible templates.You can now use .format_messages(element="...") to inject

content into {element}.

1 «#4####Chaining

2 fxom langchain_core.prompts import ChatPromptTemplate

3

4

5 prompt_text= You are a helpful assistant tasked with summarizing text.Give a concise summary of the
NarrativeText.NarrativeText {element}"""

6
prompt_table="""You are a helpful assistant tasked with summarizing tables .Give a concise summary of the
table.Table {element}"""

8
# These lines convert the plain text strings into LangChain-compatible templates.

1@ # You can now use .format_messages(element="...") to inject content into {element}.

11 prompt_text = ChatPromptTemplate.from_template(prompt_text)
12 prompt_table=ChatPromptTemplate.from_template(prompt_table)

Creating a data flow pipeline, where each component transforms the input and passes it to the next:

1. {"element": lambda x: x}:The key "element" matches the {element} placeholder in your prompt templates.

2.| prompt_text or | prompt_table:This takes the formatted dictionary and applies it to the prompt using ChatPromptTemplate.
3. | lim: This sends the prompt to your LLM (Language Model), e.g., sambanova, etc., and gets a response.

4. | StrOutputParser():This parses the output and converts it into a plain string (instead of an LLM message object).

1 text_summarize_chain = {"element": lambda x: x} | prompt_text | 11m | StrOutputParser()
2 table_summarize_chain= {"element": lambda x: x} | prompt_table | 1lm | StrOutputParser()

eo Batch() method in LangChain to summarize multiple narrative texts in parallel or controlled batches.

{'max_concurrency': 1}:his sets the maximum number of parallel executions to 1, meaning it processes one input at a time

(sequentially).

1 NarrativeText_summaries = []

2 if NarrativeText:

3 NarrativeText_summaries = text_summarize_chain.batch(NarrativeText, {'max_concurrency': 1}) #This sets the
maximum number of parallel executions to 1, meaning it processes one input at a time (sequentially).
```

![Page 5](./images_md/page_5.png)

**OCR Text for Page 5:**
```
1 table_summaries = []
2  table_summaries=table_summarize_chain.batch(table,{"max_concurrency": 3})

B]Image Summary:

Uses the SAMBANOVA API to summarize images. It takes a list of image files, encodes them as base64 strings, and then uses

the image_summarizer function to generate summaries for each image.

code snippet:

import openai

import base64

import os

from langchain_core.messages import HumanMessage

anwWwNPR

from IPython.display import HTML, display

1 client = openai.OpenAI(

2 base_url="https://api.sambanova.ai/v1",
3 api_key="SAMBANOVA_API_KEY")

1 def encode_image(image_path):

2 with open(image_path, "rb") as image_file:

3 return base64.bé4encode(image_file.read()) .decode('utf-8')
4

5 def image_summarizer(prompt, image_base64) :

6 response = client.chat.completions.create(

7 model="Llama-4-Maverick-17B-128E-Instruct",

8 messages=[

9 {
10 "role": "user",
11 "content": [
12 {"type": "text", "text": prompt}
13 {"type": "image_url", "image_url": {"url": f"data: image/jpeg; base64,{image_base64}"}}
14 ]
15 }
16 ]

17 )

18

19, return response.choices[0].message.content

1

2 def generate_img_summaries(path):

3 wan

4 Generate summaries and base64 encoded strings for images

5 path: Path to list of .jpg files extracted by Unstructured
6 wan

7

8 # Store baseé4 encoded images

9 img_base64_list = []

10

11 # Store image summaries

12 image_summaries = []

13

14 # Prompt

15 prompt = """You are an assistant tasked with summarizing images for retrieval. \

16 These summaries will be embedded and used to retrieve the raw image. \
```

![Page 6](./images_md/page_6.png)

**OCR Text for Page 6:**
```
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33

Give a concise summary of the image that is well optimized for retrieval.

# Apply to images
for img_file in sorted(os.listdixr(path)):
if img_file.endswith(".jpg"):
img_path = os.path.join(path, img_file)
base64_image = encode_image(img_path)
img_base64_list.append(base64_image)
generated_summary = image_summarizer(prompt, base64_image)
print (generated_summary)
image_summaries . append(image_summarizer(prompt ,base64_image) )

return img_base64_list, image_summaries

# Image summaries

img_base64_list, image_summaries = generate_img_summaries("/Users/shivanim/Desktop/ML_PRACTICE/image_store")
#image stored path

STEP4.Adding to the Vector Store 2

The 4 steps to create a MultiVectorRetriever involve initializing a storage tier, creating a vector store, creating a

MultiVectorRetriever instance, and adding documents to the retriever, which enables fast and semantically accurate retrieval of

text, tables, and images while preserving access to original content.

Code snippet:
1 import uuid
2 from langchain.retrievers.multi_vector import MultiVectorRetriever
3 from langchain.storage import InMemoryStore
4 from langchain_chroma import Chroma
5 from langchain_core.documents import Document
6 # from langchain_community. embeddings import SambaStudioEmbeddings
7 fxom langchain_sambanova import SambaNovaCloudEmbeddings
8 from dotenv import load_dotenv
9 load_dotenv()

import os

Create Embedding using SambaNovaCloudEmbeddings:

1
2
3

anw nr

from langchain_sambanova import SambaNovaCloudEmbeddings

embeddings = SambaNovaCloudEmbeddings(
model="E5-Mistral-7B-Instruct",sambanova_api_key="SAMBANOVA_API_KEY")

# This function creates a MultiVectorRetriever that indexes embeddings of summaries (text, table, and image)
in a vector store, while storing the original content in an in-memory docstore. When queried, the retriever
uses vector similarity on the summaries to retrieve the most relevant results and maps them back to the
original full documents. This enables fast, semantically accurate retrieval while preserving access to
detailed source data.

def create_multi_vector_retriever(vectorstore,text_summaries, texts, table_summaries, table
image_summaries, img
```

![Page 7](./images_md/page_7.png)

**OCR Text for Page 7:**
```
0 0 YO

11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
58
51
52
53
54
55
56
57
58
59

2
3

Create a retriever that indexes the summary but returns the original image or text.

# Initialize the storage tier
store = InMemoryStore()
id_key = "doc_id"

# vectorstore = Chroma(
# # collection_name='summaries',embedding_function=embeddings )
# print(vectorstore)
# print (table_summaries)
# print(texts)

# The retriever (empty to start)

retriever = MultiVectorRetriever(vectorstore=vectorstore,
docstore=store,
id_key=id_key)
# search_kwargs={'k': 2})

print (retriever)

def add_documents(retriever, doc_summaries, doc_contents):
print(doc_contents)
doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
print(retriever)
summary_docs = [
Document(page_content=s, metadata={id_key: doc_ids[i]})
for i, s in enumerate(doc_summaries)
]

print (summary_docs)

retriever.vectorstore.add_documents(summary_docs)
retriever .docstore.mset(list(zip(doc_ids, doc_contents)))

# Add texts, tables, and images

# Check that text_summaries is not empty before adding
if text_summaries:

# print (text_summaries)

add_documents(retriever, text_summaries, texts)

# Check that table_summaries is not empty before adding
if table_summaries:

# print (table_summaries)

add_documents(retriever, table_summaries, table)
# Check that image_summaries is not empty before adding
if image_summaries:

add_documents(retriever, image_summaries, img)

return retriever

# The vectorstore to use to index the child chunks
vectorstore = Chroma(

collection_name='summaries' ,embedding_function=embeddings, persist_directory="sample-rag-multi-modal" )
```

![Page 8](./images_md/page_8.png)

**OCR Text for Page 8:**
```
4 print(vectorstore)

6 # def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, table,
image_summaries, img

7 #):
# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(

10 vectorstore,

11 NarrativeText_summaries
12 NarrativeText,

ils} table_summaries,

14 table,

15 image_summaries,

16 img_base64_list,

iy ®)

18

19 retriever_multi_vector_img

STEP 4:Preparing Input Data for Multimodal Retrieval: A Key to Accurate Results 2

In the field of multimodal retrieval, it's essential to preprocess input data to prepare it for the retrieval system. This involves
splitting the input data into separate lists for images and text, resizing images to a consistent size, and formatting the input data

into a prompt for the language model.

e Preprocessing input data is crucial for multimodal retrieval because it allows the retrieval system to effectively retrieve
relevant information and generate a response to the user's question. By splitting the input data into separate lists for
images and text, resizing images to a consistent size, and formatting the input data into a prompt for the language model,
we can ensure that the retrieval system has the necessary information to provide accurate and relevant results.

Code snippet:
1 import io
2 import re
3
4 from IPython.display import HTML, display
5 from langchain_core.runnables import RunnableLambda, RunnablePassthrough
6 | from PIL import Image
1 def plt_img_base64(img_base64):
2 """Disply base64 encoded string as image"""
3 # Create an HTML img tag with the base64 string as the source
4 image_html = f'<img srce="data:image/jpeg;base64,{img_base64}" />'
5 # Display the image by rendering the HTML
6 display (HTML (image_htm1) )
7
8
9 def looks_like_base64(sb):
10 """Check if the string looks like base64"""
11) return re.match("*[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
12
13 def is_image_data(b64data):
14 uuu

15) Check if the base64 data is an image by looking at the start of the data
```

![Page 9](./images_md/page_9.png)

**OCR Text for Page 9:**
```
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
58
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
78
71
72
73

def

def

def

image_signatures = {
b"\xff\xd8\xff": "jpg",
b"\x89\x50\x4e\x47\x@d\x@a\x1la\x@a": "png",
b"\x47\x49\x46\x38": "gif",
b"\x52\x49\x46\x46": "webp",

+

try:
header = base64.b64decode(bé4data)[:8] # Decode and get the first 8 bytes
for sig, format in image_signatures.items():

if header.startswith(sig):
return True

return False

except Exception:
return False

resize_base64_image(base64_string, size=(128, 128)):

Resize an image encoded as a Base64 string
# Decode the Baseé4 string

img_data = base64.bé4decode(base64_string)
img = Image.open(io.BytesI0(img_data))

# Resize the image
resized_img = img.resize(size, Image.LANCZOS)

# Save the resized image to a bytes buffer
buffered = io.BytesI0()
resized_img.save(buffered, format=img. format)

# Encode the resized image to Base64
return base64.bé64encode(buffered. getvalue()) .decode("utf-8")

split_image_text_types(docs):

Split base64-encoded images and texts
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
texts .append(doc)
return {"images": b64_images, "texts": texts}

img_prompt_func(data_dict):

Join the context into a single string
formatted_texts = "\n".join(data_dict["context"]["texts"])
messages = []
```

![Page 10](./images_md/page_10.png)

**OCR Text for Page 10:**
```
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97

# Adding image(s) to the messages if present
if data_dict["context"]["images"]:
for image in data_dict["context"]["images"]:
image_message = {
"type": "image_url"
"image_url": {"url": f"data:image/jpeg;base64,{image}"},
+

messages .append(image_message)

# Adding the text for analysis
text_message = {
"type": "text",
"text": (
"You are analyst and advice.\n"
"You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
"Use this information to provide answer related to the user question. \n"
f"User-provided question: {data_dict['question']}\n\n"
"Text and / or tables:\n"
"{formatted_texts}"
),
+
messages ..append(text_message)
return [HumanMessage(content=messages) ]

Step 5:Multimodal RAG Chain in Information Retrieval ?

The Multimodal RAG Chain is used after retrieval to generate a response to the user's query by combining the retrieved text and

images. This chain of operations enables the system to format a multimodal prompt and pass it to the LLM for processing. The

resulting response is then parsed into a plain string, providing a final answer to the user's question

Code snippet:

q
2
3
4
5
6
7
8
9

18
11)
12
13
14
15)
16
17
18
19
28
21)

def multi_modal_rag_chain(retriever) :

Multi-modal RAG chain

# Multi-modal LLM
model = ChatSambaNovaCloud(model="Llama-4-Maverick-17B-128E-Instruct", temperature=0.7,top_p=0.01)
# RAG pipeline
chain = (
{

"context": retriever | RunnableLambda(split_image_text_types) ,

"question": RunnablePassthrough(),
+

| RunnableLambda(img_prompt_func)

| model

| StrOutputParser()

return chain
```

![Page 11](./images_md/page_11.png)

**OCR Text for Page 11:**
```
22 chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

STEP 6:Querying the Multimodal RAG Chain 2

The multimodal RAG chain can generate different types of responses depending on the query and the data stored in the vector

store. There are three main types of responses:

Code snippet:

Typel:Multimodal Response (Image and Text): ?

This type of response includes both text and images, retrieved from the vector store, in response to a user's query. The

query_multimodal_rag function generates this type of response.

1 # ---Text and Image---

2

3 def query_multimodal_rag(query_text) :

4 response = chain_multimodal_rag.invoke(query_text)

5 print(response)

6

7 # Get docs separately (already retrieved)

8 docs = retriever_multi_vector_img.invoke(query_text, limit= 6)
9

10 # Display all images retrieved and detected as base64 images
11 for img in split_image_text_types(docs)["images"]

12 plt_img_base64 (img)

13)

14 # User can enter the query
15 query_text = input("Enter your query: ")
16 query_multimodal_rag(query_text)

Type2:Image-Only Response: 2

This type of response only includes images, retrieved from the vector store, in response to a user's query. The code snippet that

queries by image and displays the retrieved images using IPyImage generates this type of response.

1 # --- Query by Image ---

2 # The response provides a visual summary of the information related to the query
3 import base64

4 from io import BytesI0

5 from IPython.display import display, Image as IPyImage

6 import re

7

8 chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

9 query_text = "What is mult-head-attention?"

18

11 docs = retriever_multi_vector_img.invoke(query_text, limit=6)

12

13 def is_base64_image(text) :

14 return isinstance(text, str) and re.match(r'4/9j|*iVB', text.strip()) # JPEG or PNG signatures
15)

16 for i, doc in enumerate(docs):
```

![Page 12](./images_md/page_12.png)

**OCR Text for Page 12:**
```
17 print(f"{i}:", end="
18

19 if is_base64_image(doc):

20 try:

21 # Display inline image

22 image_data = base64.bé4decode(doc)

23 display (IPyImage(data=image_data) )

24 except Exception as e:

25 print(f"[Exror displaying image]: {e}")
26 else:

27 print (doc)

28 print(docs)

Type3:Text-Only Response: ?

This type of response only includes text, retrieved from the vector store, in response to a user's query. The code snippet that

queries by text and prints the response using print(xesponse_text) generates this type of response.
# --- Query by TEXT ---
query_text = "What is mult-head-attention?"

response_text = chain_multimodal_rag.invoke(query_text)
print("Response to text query:")

Yon RnAnP

print (response_text)

For your reference here are the notebook and and sample pdf docs please refer to github

By following these steps, Multimodal RAG can provide a more accurate and relevant response to the user's query, and improve

the overall user experience.

Thank you and happy learning.
```

