from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticMatcher:
    def __init__(self):
        # We use a lightweight local embedding model
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def compute_semantic_match(self, jd_text: str, resume_text: str) -> str:
        """
        Calculates semantic match using ChromaDB.
        Embeds the resume, then queries the JD against the resume text.
        Returns a context string of highly matched areas.
        """
        try:
            jd_chunks = self.text_splitter.split_text(jd_text)
            resume_chunks = self.text_splitter.split_text(resume_text)

            if not jd_chunks or not resume_chunks:
                return "Insufficient text for mapping."

            # We embed the resume chunks into a vector store
            vectorstore = Chroma.from_texts(
                texts=resume_chunks, 
                embedding=self.embedding_function
            )

            match_results = []
            # Query each JD chunk against the resume
            for jd_chunk in jd_chunks:
                docs = vectorstore.similarity_search_with_score(jd_chunk, k=2)
                for doc, score in docs:
                    # Lower score in Chroma (L2 distance usually) means closer match
                    if score < 1.5: # threshold can be tuned
                        match_results.append(f"JD Requirement: '{jd_chunk[:100]}...' -> Candidate Evidence: '{doc.page_content}' (Distance: {score:.2f})")
            
            if not match_results:
                return "No strong semantic matches found via vector search."
                
            return "\n".join(match_results[:5]) # Top 5 semantic connections
        except Exception as e:
            return f"Error during semantic matching: {str(e)}"

# Singleton
semantic_matcher = SemanticMatcher()
