import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

llm = ChatGroq(model="llama3-8b-8192")
embed = OllamaEmbeddings(model="llama3.2:1b")

###Creating RAG pipeleine
import numpy as np

class RAGPipeline:
    def __init__(self, llm_model: str = "llama3-8b-8192" , ollama_embed_model: str = "llama3.2:1b"):
        self.llm = ChatGroq(model=llm_model)
        self.embed = OllamaEmbeddings(model=ollama_embed_model)
        self.doc_embeddings = None
        self.docs = None

    async def aload_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = await self.embed.aembed_documents(self.docs)
        return self.docs, self.doc_embeddings

    async def aget_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")
        
        query_embedding = await self.embed.aembed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]
    
    async def agenerate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = await self.llm.ainvoke(messages)
        return ai_msg

# # Sample usage
# if __name__ == "__main__":
#     import asyncio

#     sample_docs = [
#         "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
#         "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
#         "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
#         "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
#         "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
#     ]

#     async def main():
#         rag = RAGPipeline()
#         await rag.aload_documents(sample_docs)
#         query = "Who introduced the theory of relativity?"
#         relevant_doc = await rag.aget_most_relevant_docs(query)
#         answer = await rag.agenerate_answer(query, relevant_doc)
#         print(f"Query: {query}")
#         print(f"Relevant Document: {relevant_doc}")
#         print(f"Answer: {answer.content}")

#     asyncio.run(main())

# Result
# Query: Who introduced the theory of relativity?
# Relevant Document: ['Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.']
# Answer: According to the document, Albert Einstein introduced the theory of relativity.
sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]

sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]


async def dataset():
    rag = RAGPipeline()
    await rag.aload_documents(sample_docs)
    dataset = []
    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = await rag.aget_most_relevant_docs(query)
        response = await rag.agenerate_answer(query, relevant_docs)
        response = response.content
        dataset.append(
            {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
        )
    return dataset

from ragas import evaluate
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

async def eval_rag():
    data = await dataset()
    evaluation_dataset = EvaluationDataset.from_list(data)
    evaluator_llm = LangchainLLMWrapper(llm)
    result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    return result

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(eval_rag())
    print(results)

# 'context_recall': 1.0000, 'faithfulness': 0.8000, 'factual_correctness(mode=f1)': 1.0000}
    
   


