# Building a Multi-Document Reader and Chatbot With LangChain and large language model 
Large language models can be inconsistent. Sometimes they nail the answer to questions, other times they regurgitate random facts from their training data. If they occasionally sound like they have no idea what they’re saying, it’s because they don’t. LLMs know how words relate statistically, but not what they mean.

Retrieval-augmented generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information. Implementing RAG in an LLM-based question answering system has two main benefits: It ensures that the model has access to the most current, reliable facts, and that users have access to the model’s sources, ensuring that its claims can be checked for accuracy and ultimately trusted.

In simple terms, RAG is to LLMs what an open-book exam is to humans. In an open-book exam, students are allowed to bring reference materials, such as textbooks or notes, which they can use to look up relevant information to answer a question. The idea behind an open-book exam is that the test focuses on the students’ reasoning skills rather than their ability to memorize specific information.

Similarly, the factual knowledge is separated from the LLM’s reasoning capability and stored in an external knowledge source, which can be easily accessed and updated:
1. Parametric knowledge: Learned during training that is implicitly stored in the neural network's weights.
2. Non-parametric knowledge: Stored in an external knowledge source, such as a vector database.

The vanilla RAG workflow is illustrated below:

![image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kSkeaXRvRzbJ9SrFZaMoOg.png)

1. Retrieve: The user query is used to retrieve relevant context from an external knowledge source. For this, the user query is embedded with an embedding model into the same vector space as the additional context in the vector database. This allows to perform a similarity search, and the top k closest data objects from the vector database are returned.
2. Augment: The user query and the retrieved additional context are stuffed into a prompt template.
3. Generate: Finally, the retrieval-augmented prompt is fed to the LLM.

In this project i am going to show you how you can chat with NVIDIA documentation by using RAG. For this i will mostly use `langchain` whitch is famouse python library and for LLM i am goinr to use `GooglePalm`. I know its not the best llm out there but you can change the model and the process is almost same except what kind of large language model you are using.

#### The library i am going are-
        
        langchain==0.0.340
        tiktoken==0.5.1
        faiss-cpu==1.7.4
        protobuf~=3.19.0
        InstructorEmbedding
        sentence-transformers
        pandas
        streamlit
        altair==4
        google-generativeai

First you need to install the libraries. Make sure you are installing all this on a virtual enviroment. The `langchaing` library will help me to intregation the whole process. I am going to use `fais` for vector database to store the embedding of the document and serch for similar content. For embedding i am going to use `sentence` transformers library and `InstrictEmbeddding` from `HiggingFace`. For visual implementation i am going to use streamlit for chat because its easy to use and looks good.
Next i am going to import all the library i needed-

        from langchain.llms import GooglePalm
        from langchain.chains import RetrievalQA
        from langchain.embeddings import GooglePalmEmbeddings
        from langchain.llms import GooglePalm
        import pandas as pd
        from langchain.document_loaders.csv_loader import CSVLoader
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        import streamlit as st

First i need to collect the api key to use the googlepalm model. After collect them i need to add them to my script and then load the model using `GooglePalm` module. I set temperite low because i dont want my llm to be very generative by its own otherwise it will start to hallucunate.


        from langchain.llms import GooglePalm
        api_key = 'api-key' 
        llm = GooglePalm(google_api_key=api_key, temperature=0.1)

Next with the help of `CSVLoader` i load the dataset from the directory. I use CSVLoader because my data is in csv format. If you use other format of dataset the you have to use different forms of loader and `langchain` has different kinds of loader.

        loader = CSVLoader(file_path='/content/NvidiaDocumentationQandApairs.csv')
        data = loader.load()

After that i am going to load the embedding model to create the embeddings from the dataset. For this i am going to use `HuggingFaceInstructEmbeddings` module `instructor-large` and the model is from sentence transformers.

        instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

Then i create a vectordatabase and a retriver to retrive the data from database. I use `faiss` to store the data and rtrive the data. after createing the database i save the database to my directory so that i can also use them for better experience.

        vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
        retriever = vectordb.as_retriever(score_threshold = 0.7)
        vectordb.save_local("nvidia_doc")

we are done with the creative part now we can start to query the database. we can use similarity search to query the data.

        new_db = FAISS.load_local("/content/nvidia_doc", instructor_embeddings)
        docs = new_db.similarity_search("What is the purpose of NVIDIA GPU Cloud (NGC)?")


But for better query result we need to use llm. Wright now we will create a prompt template to give a proper instruction to our llm how to act as a assistant to get better quality of information with the help of `PromptTemplate`. Then we will create a chain using `RetrievalQA`.

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
        
        CONTEXT: {context}
        
        QUESTION: {question}"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        
        chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs)


Now we can start our query to retrive the desired informatiom. 

For more interactive visualization we will use streamlit. this will create a beautidul chat platform to communicate with the data.

        st.title("Question and Answer with NVIDIA documentation")
        
        
        question = st.text_input("Question: ")
        
        if question:
            chain = get_qa_chain()
            response = chain(question)
        
            st.header("Answer")
            st.write(response["result"])

