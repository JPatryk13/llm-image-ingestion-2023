from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pickle


def query_data(*, vectorstore: str, open_api_key: str, query: str) -> None:

    with open(vectorstore, "rb") as f:
        vectorstore = pickle.load(f)

    llm = OpenAI(model_name="text-davinci-003", openai_api_key=open_api_key)

    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa({"query": query})

    print(result['result'])


if __name__ == '__main__':
    # Works fine for .txt and .pdf, start hallucinating for .png. Seems to read the image within the pdf doc, might be
    # cheating though and infer the info from the text
    query_data(
        vectorstore="vectorstores/vectorstore-from-img-png.pkl",
        open_api_key='...',
        query="What can you tell me about the graph within the document?"
    )