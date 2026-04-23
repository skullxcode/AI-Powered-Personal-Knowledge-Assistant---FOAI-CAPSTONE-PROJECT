from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
import os
from typing import Optional, List, Any

_llm_instance = None
_current_token = None

class CustomOnlineLLM(LLM):
    client: Any = None

    def __init__(self, token: str = "", **kwargs):
        super().__init__(**kwargs)
        # Using the exact API approach provided by the user
        hf_token = token if token else os.environ.get("HF_TOKEN", "")
        self.client = InferenceClient(api_key=hf_token)

    @property
    def _llm_type(self) -> str:
        return "custom_online_hf"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        completion = self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=600,
        )
        return completion.choices[0].message.content

def get_llm(token: str = ""):
    global _llm_instance, _current_token
    if _llm_instance is None or (token and token != _current_token):
        print("Initializing Online Inference LLM...")
        _llm_instance = CustomOnlineLLM(token=token)
        if token:
            _current_token = token
    return _llm_instance

def create_conversational_chain(vector_db, memory, token: str = ""):
    llm = get_llm(token)
    
    template = """You are a highly capable analytical assistant. 
Answer the question relying ONLY on the Context provided below. If you cannot extract the answer from the Context, state 'I cannot answer based on the provided documents.'

Context:
{context}

Question:
{question}
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def summarize_document(document_content, token: str = ""):
    llm = get_llm(token)
    # Providing a distinct summarization instruction to SmolLM
    prompt = f"Provide a comprehensive high-level summary of the following document:\n\n{document_content[:2500]}"
    return llm.invoke(prompt)
