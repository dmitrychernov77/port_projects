from typing import List, Any

from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.readers import SimpleWebPageReader


class RAG:
    def __init__(self, urls: List[str], **kwargs: Any) -> None:
        self.model = kwargs.get("model", "gpt-4")
        self.temperature = kwargs.get("temperature", 0.0)
        self.chunk_size = kwargs.get("chunk_size", 512)
        self.chunk_overlap = kwargs.get("chunk_overlap", 64)
        self.similarity_top_k = kwargs.get("similarity_top_k", 5)

        self.llm = OpenAI(model=self.model,
                          temperature=self.temperature
                          )
        service_context = ServiceContext.from_defaults(llm=self.llm)

        self.documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        node_parser = SimpleNodeParser.from_defaults(chunk_size=self.chunk_size, 
                                                     chunk_overlap=self.chunk_overlap)
        self.nodes = node_parser.get_nodes_from_documents(self.documents)
        self.index = VectorStoreIndex(self.nodes,
                                      service_context = service_context)

        self.query_engine = self.index.as_query_engine(similarity_top_k=self.similarity_top_k)

    def __call__(self, query: str) -> str:
        response = self.query_engine.query(query)
        return response.response


if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/C%2B%2B",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
    ]
    rag = RAG(urls)

    response = rag("Which programming language is faster: Python, C++ or Rust?")
    print(response)

    response = rag("When Python was created?")
    print(response)

    response = rag("Is python easy to learn? Give a short answer")
    print(response)
