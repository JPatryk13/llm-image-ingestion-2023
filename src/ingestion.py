from langchain.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, ImageCaptionLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


def ingest_docs(*, filename: str, openai_api_key: str) -> None:
    """
    Taking in files, splitting data into chunks, vectorising and saving locally.

    :param filename:
    :param openai_api_key:
    :return:
    """

    file_ext = filename[filename.find('.'):]

    # Some simple example file extensions for now
    if '.txt' in file_ext:
        loader = TextLoader(filename)
    if '.docx' in file_ext:
        # TODO:
        #  Crashes at:
        #  KeyError: "no relationship of type 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' in collection"
        loader = UnstructuredWordDocumentLoader(filename)
    if '.pdf' in file_ext:
        loader = PyPDFLoader(filename)
    if '.png' in file_ext:
        loader = ImageCaptionLoader(filename)
    else:
        # handling any edge cases?
        pass

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = FAISS.from_documents(docs, embeddings)

    # Save vectorstore
    with open(f"vectorstores/vectorstore-from-{filename.replace('.', '-')}.pkl", "wb") as f:
        pickle.dump(db, f)


if __name__ == "__main__":
    ingest_docs(
        filename='...',
        openai_api_key='...'
    )
