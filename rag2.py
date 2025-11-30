# rag_engine.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class MBA_RAG_Engine:
    def __init__(self, db_folder="chroma_db"):
        self.db_folder = db_folder
        os.makedirs(self.db_folder, exist_ok=True)

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3
        )

        # Prompt template
        template = """
Role:
You are an MBA Case Debrief Bot trained to analyze business case studies.

Content Type:
Structured, concise, MBA-style analysis.

Context:
Use ONLY the retrieved case data provided.
Do NOT add external knowledge.

Question: {question}
Context: {context}

Do’s:
- Be factual
- Be concise
- Follow MBA consulting frameworks
- Provide point-wise answers
- If recommendation → include rationale + action plan
- If the answer is not in the context, say "The case document does not provide this information."

Don’ts:
- Do NOT hallucinate
- Do NOT invent data not found in the case

Format:
- Bullet points
- Short explanations
"""

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    def build_vectorstore(self, docs):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=self.db_folder
        )
        return vectorstore

    def create_qa_chain(self, vectorstore):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )

    def answer_question(self, qa_chain, query):
        """Query RAG chain and return final answer."""
        result = qa_chain.invoke({"query": query})
        return result["result"]
