import logging
import asyncio
import os
import glob
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from ollama import chat
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document as LCDocument
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Setup logging with more detail
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Document:
    """Document class to store content, metadata, and hierarchy info."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class LlamaBot:
    def __init__(
        self, 
        model_name: str = "llama3.1:8b", 
        embedding_model: str = "llama3.1:8b",
        max_history: int = 10,
        system_prompt: str = "You are Stark, an AI assistant, responding only in English with concise replies.",
        data_dir: str = "./data",
        vector_store_path: str = "./vector_db",
        metadata_path: str = "./metadata"
    ):
        """
        Initialize the LlamaBot with hierarchical PDF RAG capabilities.
        
        Args:
            model_name: The Ollama model to use for chat (default: "llama3.1:8b")
            embedding_model: The Ollama model to use for embeddings (default: "llama3.1:8b")
            max_history: Maximum number of messages to keep in history per user (default: 10)
            system_prompt: System prompt template to use for chat
            data_dir: Directory containing PDF documents to load
            vector_store_path: Directory to store vector embeddings
            metadata_path: Directory to store document metadata
        """
        self.user_chat_histories = {}
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.data_dir = data_dir
        self.vector_store_path = vector_store_path
        self.metadata_path = metadata_path
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        os.makedirs(metadata_path, exist_ok=True)
        
        # Initialize RAG components
        self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.chunks = []
        self.document_hierarchy = {}  # Store hierarchical information about documents
        
        # Initialize and load documents
        self._initialize_rag()
        
        logger.info(f"LlamaBot initialized with model: {model_name}")
    
    def _initialize_rag(self):
        """Initialize RAG components: load PDFs, create embeddings, and setup vector store."""
        try:
            # Step 1: Load PDF documents
            self.documents = self._load_pdf_documents(self.data_dir)
            if not self.documents:
                logger.warning(f"No PDF documents found in {self.data_dir}")
                self._create_sample_document()  # Create a sample document if none found
                # Try loading again with the sample document
                self.documents = self._load_pdf_documents(self.data_dir)
                if not self.documents:
                    logger.warning("Still no documents after creating sample, RAG will not be available")
                    return
            
            logger.info(f"Loaded {len(self.documents)} PDF documents")
            
            # Step 2: Split documents into hierarchical chunks
            self.chunks, self.document_hierarchy = self._split_documents_hierarchically(self.documents)
            logger.info(f"Created {len(self.chunks)} chunks with hierarchical structure")
            
            # Save the document hierarchy for future use
            self._save_document_hierarchy()
            
            # Step 3: Initialize embeddings
            embedding_init_success = False
            try:
                logger.info(f"Initializing embeddings with model: {self.embedding_model}")
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
                
                # Test the embeddings to make sure they work
                test_embed = self.embeddings.embed_query("Test query")
                if isinstance(test_embed, list) and len(test_embed) > 0:
                    logger.info(f"Embeddings initialized successfully with dimension {len(test_embed)}")
                    embedding_init_success = True
                else:
                    logger.error("Embeddings test failed - returned empty or invalid embedding")
            except Exception as e:
                logger.error(f"Error initializing embeddings: {str(e)}")
            
            # Step 4: Create vector store with FAISS (only if embeddings are working)
            if self.chunks and embedding_init_success:
                # Check if a saved vector store exists
                if os.path.exists(f"{self.vector_store_path}.faiss"):
                    try:
                        # Load existing vector store
                        logger.info(f"Loading existing FAISS vector store from {self.vector_store_path}")
                        self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
                        logger.info(f"Loaded existing FAISS vector store from {self.vector_store_path}")
                    except Exception as e:
                        logger.error(f"Error loading FAISS vector store: {str(e)}")
                        # Create a new vector store if loading fails
                        self._create_new_vector_store()
                else:
                    # Create a new vector store
                    logger.info("No existing vector store found, creating new one")
                    self._create_new_vector_store()
            elif self.chunks:
                # If embeddings failed but we have chunks, create keyword index as fallback
                logger.warning("Embeddings initialization failed, creating keyword index as fallback")
                self._create_keyword_index()
                logger.info("Keyword index created successfully")
            else:
                logger.warning("No chunks to add to vector store")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            # Create fallback keyword index in case of failure
            if self.chunks:
                self._create_keyword_index()
    
    def _create_sample_document(self):
        """Create a sample document for testing when no PDFs are found."""
        try:
            # Create a simple text file with some content
            sample_path = os.path.join(self.data_dir, "sample.txt")
            with open(sample_path, "w") as f:
                f.write("# Sample Document\n\n")
                f.write("This is a sample document for testing purposes.\n\n")
                f.write("## Section 1: Introduction\n\n")
                f.write("This section provides an introduction to the topic.\n\n")
                f.write("## Section 2: Main Content\n\n")
                f.write("This is the main content of the sample document.\n\n")
                f.write("## Section 3: Conclusion\n\n")
                f.write("This section concludes the document.\n")
            
            logger.info(f"Created sample document at {sample_path}")
        except Exception as e:
            logger.error(f"Error creating sample document: {str(e)}")
    
    def _create_new_vector_store(self):
        """Create a new FAISS vector store from chunks."""
        try:
            # Convert our Document objects to LangChain documents
            logger.info(f"Creating new vector store with {len(self.chunks)} chunks")
            lc_docs = []
            for doc in self.chunks:
                lc_docs.append(LCDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ))
            
            # Create vector store
            self.vector_store = FAISS.from_documents(lc_docs, self.embeddings)
            
            # Save vector store to disk
            self.vector_store.save_local(self.vector_store_path)
            
            logger.info(f"Created and saved new FAISS vector store with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error creating new FAISS vector store: {str(e)}")
            # Create a simpler keyword-based index as fallback
            self._create_keyword_index()
    
    def _create_keyword_index(self):
        """Create a simple keyword index when vector embeddings fail."""
        try:
            # Create a simple dictionary-based index
            self.keyword_index = {}
            if not self.chunks:
                logger.warning("No chunks available to create keyword index")
            return
            
            logger.info(f"Creating keyword index from {len(self.chunks)} chunks")
            for i, doc in enumerate(self.chunks):
                # Extract words and add to index
                words = set()
            
            # Simple text normalization and tokenization
            text = doc.page_content.lower()
            # Remove common punctuation
            for char in ".,;:!?()[]{}\"'":
                text = text.replace(char, " ")
            
            # Split by whitespace and add to set
            for word in text.split():
                if len(word) > 2:  # Skip very short words
                    words.add(word)
            
            # Add document index to each word's posting list
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append(i)
        
            logger.info(f"Created fallback keyword index with {len(self.keyword_index)} terms")
        except Exception as e:
            logger.error(f"Error creating fallback keyword index: {str(e)}")
            # Initialize empty index as last resort
            self.keyword_index = {}
    
    def _keyword_search(self, query: str, k: int = 4) -> List[Document]:
        """Simple keyword-based search when vector search is unavailable."""
        if not self.chunks:
            return []
        
        if hasattr(self, 'keyword_index') and self.keyword_index:
            # Use the keyword index
            query_terms = query.lower().split()
            doc_scores = {}
            
            for term in query_terms:
                if term in self.keyword_index:
                    for doc_idx in self.keyword_index[term]:
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
            
            # Sort by score and get top k
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            return [self.chunks[idx] for idx, _ in sorted_docs]
        else:
            # Fallback to simple scan
            query_terms = query.lower().split()
            scores = []
            
            for doc in self.chunks:
                content = doc.page_content.lower()
                score = sum(content.count(term) for term in query_terms)
                scores.append(score)
            
            # Get top k indices
            if not scores:
                return []
                
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self.chunks[i] for i in top_indices]
    
    def _load_pdf_documents(self, directory: str) -> List[Document]:
        """
        Load PDF documents from a directory.
        
        Args:
            directory: Directory containing PDF files
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Get all PDF files in the directory
            pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
            
            if not pdf_files:
                # Try looking for text files as well
                txt_files = glob.glob(os.path.join(directory, "*.txt"))
                if txt_files:
                    logger.info(f"Found {len(txt_files)} text files in {directory}")
                    for txt_path in txt_files:
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            doc_id = str(uuid.uuid4())
                            filename = os.path.basename(txt_path)
                            
                            # Create document with metadata
                            doc_metadata = {
                                "doc_id": doc_id,
                                "filename": filename,
                                "source": txt_path,
                                "type": "text",
                                "total_pages": 1
                            }
                            
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    **doc_metadata,
                                    "hierarchy_level": "page",
                                    "hierarchy_id": f"{doc_id}_page_1",
                                    "parent_id": doc_id,
                                    "page": 1
                                }
                            ))
                            
                            logger.info(f"Loaded text file: {filename}")
                        except Exception as e:
                            logger.error(f"Error loading text file {txt_path}: {str(e)}")
                    
                    # Return the loaded text documents
                    return documents
                
                logger.warning(f"No PDF or text files found in {directory}")
                return documents
            
            # Process each PDF file
            for pdf_path in pdf_files:
                try:
                    # Load the PDF using PyPDFLoader
                    loader = PyPDFLoader(pdf_path)
                    pdf_docs = loader.load()
                    
                    doc_id = str(uuid.uuid4())
                    filename = os.path.basename(pdf_path)
                    
                    # Store basic metadata for the document
                    doc_metadata = {
                        "doc_id": doc_id,
                        "filename": filename,
                        "source": pdf_path,
                        "type": "pdf",
                        "total_pages": len(pdf_docs)
                    }
                    
                    # Convert to our Document format with enhanced metadata
                    for i, doc in enumerate(pdf_docs):
                        doc_meta = dict(doc.metadata)
                        doc_meta.update(doc_metadata)
                        doc_meta["hierarchy_level"] = "page"
                        doc_meta["hierarchy_id"] = f"{doc_id}_page_{i+1}"
                        doc_meta["parent_id"] = doc_id
                        
                        documents.append(Document(
                            page_content=doc.page_content,
                            metadata=doc_meta
                        ))
                    
                    logger.info(f"Loaded PDF: {filename} with {len(pdf_docs)} pages")
                    
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF documents: {str(e)}")
            return []
    
    def _split_documents_hierarchically(self, documents: List[Document]) -> Tuple[List[Document], Dict]:
        """
        Split documents into hierarchical chunks using recursive text splitters.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (chunks list, hierarchy dictionary)
        """
        chunks = []
        document_hierarchy = {}
        
        try:
            # Group documents by their parent document ID
            docs_by_parent = {}
            for doc in documents:
                parent_id = doc.metadata.get("parent_id")
                if parent_id not in docs_by_parent:
                    docs_by_parent[parent_id] = []
                docs_by_parent[parent_id].append(doc)
            
            # For each parent document, process its pages hierarchically
            for parent_id, parent_docs in docs_by_parent.items():
                document_hierarchy[parent_id] = {
                    "id": parent_id,
                    "filename": parent_docs[0].metadata.get("filename", ""),
                    "total_pages": len(parent_docs),
                    "sections": {},
                    "chunks": []
                }
                
                # Sort pages by page number
                parent_docs.sort(key=lambda x: x.metadata.get("page", 0))
                
                # Process each page
                for page_doc in parent_docs:
                    page_num = page_doc.metadata.get("page", 0)
                    
                    # First split into sections (headers and content blocks)
                    section_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ". ", " ", ""],
                        chunk_size=800,
                        chunk_overlap=100
                    )
                    
                    # Convert our Document to LangChain document for splitting
                    lc_page_doc = {"page_content": page_doc.page_content, "metadata": page_doc.metadata}
                    
                    # Keep section metadata
                    section_texts = section_splitter.split_text(page_doc.page_content)
                    section_chunks = []
                    
                    # Create section documents with metadata
                    for i, section_text in enumerate(section_texts):
                        section_metadata = dict(page_doc.metadata)  # Copy page metadata
                        section_id = f"{parent_id}_page_{page_num}_section_{i+1}"
                        section_metadata["hierarchy_level"] = "section"
                        section_metadata["hierarchy_id"] = section_id
                        section_metadata["section_num"] = i + 1
                        
                        section_chunks.append({
                            "page_content": section_text,
                            "metadata": section_metadata
                        })
                    
                    # Add sections to hierarchy
                    if str(page_num) not in document_hierarchy[parent_id]["sections"]:
                        document_hierarchy[parent_id]["sections"][str(page_num)] = []
                    
                    # Further split sections into smaller chunks for retrieval
                    paragraph_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n", ". ", " ", ""],
                        chunk_size=200,
                        chunk_overlap=50
                    )
                    
                    for i, section in enumerate(section_chunks):
                        section_id = section["metadata"]["hierarchy_id"]
                        
                        # Add section to hierarchy
                        document_hierarchy[parent_id]["sections"][str(page_num)].append({
                            "id": section_id,
                            "section_num": i + 1,
                            "content_preview": section["page_content"][:100] + "...",
                            "chunks": []
                        })
                        
                        # Create smaller chunks from the section
                        para_texts = paragraph_splitter.split_text(section["page_content"])
                        
                        # Update chunk metadata and add to results
                        for j, chunk_text in enumerate(para_texts):
                            chunk_id = f"{section_id}_chunk_{j+1}"
                            chunk_metadata = dict(section["metadata"])  # Copy section metadata
                            chunk_metadata["hierarchy_level"] = "chunk"
                            chunk_metadata["hierarchy_id"] = chunk_id
                            chunk_metadata["parent_section"] = section_id
                            chunk_metadata["chunk_num"] = j + 1
                            
                            # Add to document hierarchy
                            section_idx = len(document_hierarchy[parent_id]["sections"][str(page_num)]) - 1
                            document_hierarchy[parent_id]["sections"][str(page_num)][section_idx]["chunks"].append(chunk_id)
                            
                            # Add to overall document chunks list
                            document_hierarchy[parent_id]["chunks"].append(chunk_id)
                            
                            # Convert to our Document class
                            chunks.append(Document(
                                page_content=chunk_text,
                                metadata=chunk_metadata
                            ))
            
            return chunks, document_hierarchy
            
        except Exception as e:
            logger.error(f"Error splitting documents hierarchically: {str(e)}")
            # If we fail to split hierarchically, create a simple flat structure
            chunks = documents
            document_hierarchy = {
                doc.metadata.get("parent_id", str(uuid.uuid4())): {
                    "id": doc.metadata.get("parent_id", str(uuid.uuid4())),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "total_pages": 1,
                    "sections": {},
                    "chunks": []
                } for doc in documents
            }
            return chunks, document_hierarchy
    
    def _save_document_hierarchy(self):
        """Save the document hierarchy to a JSON file for persistence."""
        try:
            hierarchy_path = os.path.join(self.metadata_path, "document_hierarchy.json")
            with open(hierarchy_path, 'w') as f:
                json.dump(self.document_hierarchy, f, indent=2)
            logger.info(f"Document hierarchy saved to {hierarchy_path}")
        except Exception as e:
            logger.error(f"Error saving document hierarchy: {str(e)}")
    
    def _load_document_hierarchy(self):
        """Load the document hierarchy from a JSON file."""
        try:
            hierarchy_path = os.path.join(self.metadata_path, "document_hierarchy.json")
            if os.path.exists(hierarchy_path):
                with open(hierarchy_path, 'r') as f:
                    self.document_hierarchy = json.load(f)
                logger.info(f"Document hierarchy loaded from {hierarchy_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading document hierarchy: {str(e)}")
            return False
    
    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        """
        Get chat history for a specific user, or create a new one if it doesn't exist.
        
        Args:
            user_id: The unique identifier for the user
            
        Returns:
            The chat history for the user
        """
        if user_id not in self.user_chat_histories:
            # Initialize with empty history
            self.user_chat_histories[user_id] = []
        
        return self.user_chat_histories[user_id]
    
    async def _modify_query(self, query: str) -> str:
        """
        Modify the user query to improve retrieval results.
        
        Args:
            query: The original user query
            
        Returns:
            Modified query optimized for retrieval
        """
        history_context = ""
        if user_id:
            history = self.get_history(user_id)
            if history:
                history_context = "Previous conversation:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in history
                )

        messages = [
            {"role": "system", "content": "You are a query refinement assistant. Enhance the query by:"
            "\n- Identifying key concepts and expanding query terms"
            "\n- Using context from previous conversation if available"
            "\n- Keeping the modified query concise and focused"
            "\n- Adding synonyms for important terms"},
            {"role": "user", "content": f"{history_context}\n\nCurrent query: {query}\nPlease enhance this query."}
        ]
        
        try:
            # Run Ollama in an executor to prevent blocking
            def run_ollama():
                response = chat(self.model_name, messages=messages)
                return response["message"]["content"]
            
            modified_query = await asyncio.get_event_loop().run_in_executor(None, run_ollama)
            logger.info(f"Modified query: '{query}' -> '{modified_query}'")
            return modified_query
            
        except Exception as e:
            logger.error(f"Error modifying query: {str(e)}")
            return query  # Fallback to original query if modification fails
    
    def _retrieve_documents_hierarchical(self, query: str, max_chunks_per_level: Dict[str, int] = None) -> List[Document]:
        """
        Retrieve relevant documents with a hierarchical approach.
        
        Args:
            query: The query to search for
            max_chunks_per_level: Maximum number of chunks to retrieve per hierarchy level
                e.g., {"page": 3, "section": 5, "chunk": 10}
            
        Returns:
            List of relevant Document objects ordered by hierarchy
        """
        if not max_chunks_per_level:
            max_chunks_per_level = {"page": 3, "section": 5, "chunk": 10}
        
        # First check if we have any chunks at all
        if not self.chunks:
            logger.warning("No document chunks available for retrieval")
            return []
        
        # Check if we have a vector store
        if not self.vector_store:
            logger.warning("Vector store not initialized, falling back to keyword search")
            # Check if we have a keyword index
            if hasattr(self, 'keyword_index') and self.keyword_index:
                keyword_results = self._keyword_search(query, max_chunks_per_level.get("chunk", 10))
                logger.info(f"Keyword search returned {len(keyword_results)} documents")
                return keyword_results
            else:
                # If no keyword index, create one now
                logger.info("Creating keyword index for search")
                self._create_keyword_index()
                keyword_results = self._keyword_search(query, max_chunks_per_level.get("chunk", 10))
                logger.info(f"Keyword search returned {len(keyword_results)} documents")
                return keyword_results
        
        try:
            # First retrieve the most relevant chunks
            k = max_chunks_per_level.get("chunk", 10)
            # FAISS similarity search
            retrieved_docs = self.vector_store.similarity_search(query, k=k)
            
            if not retrieved_docs:
                logger.warning("Vector search returned no results, trying keyword search")
                return self._keyword_search(query, k)
            
            # Convert LangChain documents back to our Document class
            chunk_docs = []
            for doc in retrieved_docs:
                chunk_docs.append(Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata
                ))

            section_ids = set()
            page_ids = set()
            result_docs = []
            
            # Collect section and page IDs
            for doc in chunk_docs:
                hierarchy_level = doc.metadata.get("hierarchy_level", "")
                
                if hierarchy_level == "chunk":
                    section_id = doc.metadata.get("parent_section", "")
                    if section_id:
                        section_ids.add(section_id)
                
                page = doc.metadata.get("page", None)
                parent_id = doc.metadata.get("parent_id", "")
                if page is not None and parent_id:
                    page_ids.add(f"{parent_id}_page_{page}")
            
            # Get full sections for context
            section_chunks = []
            for i, doc in enumerate(self.chunks):
                hierarchy_id = doc.metadata.get("hierarchy_id", "")
                if hierarchy_id in section_ids:
                    section_chunks.append(doc)
                    if len(section_chunks) >= max_chunks_per_level.get("section", 5):
                        break
            
            # Get page overviews for higher-level context
            page_chunks = []
            for i, doc in enumerate(self.documents):
                hierarchy_id = doc.metadata.get("hierarchy_id", "")
                if hierarchy_id in page_ids:
                    page_chunks.append(doc)
                    if len(page_chunks) >= max_chunks_per_level.get("page", 3):
                        break
            
            # Order results by hierarchy level (page > section > chunk)
            for doc in page_chunks:
                doc.metadata["retrieval_level"] = "page"
                result_docs.append(doc)
            
            for doc in section_chunks:
                doc.metadata["retrieval_level"] = "section"
                result_docs.append(doc)
            
            for doc in chunk_docs:
                doc.metadata["retrieval_level"] = "chunk"
                result_docs.append(doc)
            
            logger.info(f"Retrieved {len(result_docs)} documents hierarchically")
            return result_docs
        
        except Exception as e:
            logger.error(f"Error retrieving documents hierarchically: {str(e)}")
            # Fallback to keyword search if vector search fails
            logger.info("Falling back to keyword search due to error")
            # Make sure we have a keyword index
            if not hasattr(self, 'keyword_index') or not self.keyword_index:
                logger.info("Creating keyword index for fallback search")
                self._create_keyword_index()
            return self._keyword_search(query, max_chunks_per_level.get("chunk", 10))
    
    def _hierarchy_aware_reranking(self, query: str, documents: List[Document]) -> List[int]:
        """
        Re-rank documents using BM25 algorithm with hierarchy awareness.
        
        Args:
            query: The query to use for ranking
            documents: List of documents to rank
            
        Returns:
            List of document indices sorted by relevance (most relevant first)
        """
        if not documents:
            return []
        
        try:
            # Tokenize documents
            corpus = [doc.page_content.split() for doc in documents]
            
            # Initialize BM25
            bm25 = BM25Okapi(corpus)
            
            # Get scores
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)
            
            # Adjust scores based on hierarchy level
            for i, doc in enumerate(documents):
                hierarchy_level = doc.metadata.get("hierarchy_level", "")
                retrieval_level = doc.metadata.get("retrieval_level", "")
                
                # Apply weight adjustments based on hierarchy
                if hierarchy_level == "page":
                    scores[i] *= 0.7  # Pages provide context but may be less specific
                elif hierarchy_level == "section":
                    scores[i] *= 1.2  # Sections are good for context and relevance
                elif hierarchy_level == "chunk":
                    scores[i] *= 1.5  # Chunks should be most directly relevant
                
                # Further boost results that were retrieved at multiple levels
                if retrieval_level == "chunk" and doc.metadata.get("parent_section", "") in [d.metadata.get("hierarchy_id", "") for d in documents]:
                    scores[i] *= 1.3  # Boost chunks that belong to a retrieved section
            
            # Sort indices based on scores in descending order
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return ranked_indices
            
        except Exception as e:
            logger.error(f"Error in hierarchy-aware reranking: {str(e)}")
            return list(range(len(documents)))  # Return original order as fallback
    
    def _create_hierarchical_context(self, ranked_docs: List[Document], max_context_length: int = 2000) -> str:
        """
        Create a context string from ranked documents, organizing by hierarchy.
        
        Args:
            ranked_docs: List of Document objects ordered by relevance
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not ranked_docs:
            return ""
        
        try:
            # Group by document and organize hierarchically
            doc_groups = {}
            for doc in ranked_docs:
                parent_id = doc.metadata.get("parent_id", "")
                if parent_id not in doc_groups:
                    doc_groups[parent_id] = {
                        "filename": doc.metadata.get("filename", ""),
                        "pages": {}
                    }
                
                page = doc.metadata.get("page", 0)
                if page not in doc_groups[parent_id]["pages"]:
                    doc_groups[parent_id]["pages"][page] = []
                
                doc_groups[parent_id]["pages"][page].append(doc)
            
            # Build context string with hierarchy information
            context_parts = []
            total_length = 0
            
            for parent_id, parent_info in doc_groups.items():
                filename = parent_info["filename"]
                context_parts.append(f"Document: {filename}")
                total_length += len(context_parts[-1])
                
                # Process each page
                for page, page_docs in sorted(parent_info["pages"].items()):
                    # Sort by section/chunk number to maintain document flow
                    page_docs.sort(key=lambda x: (
                        x.metadata.get("section_num", 0),
                        x.metadata.get("chunk_num", 0)
                    ))
                    
                    context_parts.append(f"Page {page}:")
                    total_length += len(context_parts[-1])
                    
                    # Add content from each doc
                    for doc in page_docs:
                        hierarchy_level = doc.metadata.get("hierarchy_level", "")
                        
                        if hierarchy_level == "section":
                            section_num = doc.metadata.get("section_num", "")
                            context_parts.append(f"Section {section_num}:")
                        
                        context_parts.append(doc.page_content)
                        total_length += len(doc.page_content)
                        
                        # Check if we've exceeded max context length
                        if total_length >= max_context_length:
                            context_parts.append("...")
                            break
                    
                    if total_length >= max_context_length:
                        break
                
                if total_length >= max_context_length:
                    break
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating hierarchical context: {str(e)}")
            return "\n\n".join([doc.page_content for doc in ranked_docs[:5]])  # Simple fallback
    
    async def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer based on query and context.
        
        Args:
            query: The user query
            context: Retrieved and ranked context
            
        Returns:
            Generated answer
        """
        system_prompt_with_context = self.system_prompt.format(context=context)
        
        messages = [
            {"role": "system", "content": system_prompt_with_context},
            {"role": "user", "content": f"Question: {query}\n\nPlease answer based on the context provided."}
        ]
        
        try:
            # Run Ollama in an executor to prevent blocking
            def run_ollama():
                response = chat(self.model_name, messages=messages)
                return response["message"]["content"]
            
            answer = await asyncio.get_event_loop().run_in_executor(None, run_ollama)
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while generating an answer: {str(e)}"
    
    async def _regular_chat(self, history: List[Dict[str, str]]) -> str:
        """Fallback to regular chat without RAG."""
        try:
            logger.info("Using regular chat without RAG")
            def run_regular_chat():
                response = chat(self.model_name, messages=history)
                return response["message"]["content"]
            
            answer = await asyncio.get_event_loop().run_in_executor(None, run_regular_chat)
            return answer
        except Exception as e:
            logger.error(f"Error in regular chat: {str(e)}")
            return f"Sorry, I encountered an error while responding: {str(e)}"
    
    async def chat(self, user_id: str, message: str) -> str:
        """
        Process a user message through the hierarchical PDF RAG pipeline and return a response.
        
        Args:
            user_id: The unique identifier for the user
            message: The message from the user
            
        Returns:
            Response from the assistant
        """
        logger.info(f"Processing chat request from user {user_id}")
        
        # Get existing history and add user message
        history = self.get_history(user_id)
        history.append({"role": "user", "content": message})
        
        try:
            # Check if RAG is available (documents and chunks loaded)
            if not self.documents or not self.chunks:
                logger.warning("No documents or chunks available, falling back to regular chat")
                answer = await self._regular_chat(history)
                history.append({"role": "assistant", "content": answer})
                self._trim_history(user_id)
                return answer
            
            # Step 1: Modify query using chat history context
            try:
                modified_query = await self._modify_query(message, user_id)  # Pass user_id for history context
            except Exception as e:
                logger.error(f"Query modification failed: {e}, using original query")
                modified_query = message
            
            # Step 2: Retrieve relevant documents with hierarchy
            max_chunks = {"page": 2, "section": 3, "chunk": 8}
            retrieved_docs = self._retrieve_documents_hierarchical(modified_query, max_chunks)
            
            if retrieved_docs:
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
                # Step 3: Re-rank documents with hierarchy awareness
                ranked_indices = self._hierarchy_aware_reranking(modified_query, retrieved_docs)
                ranked_docs = [retrieved_docs[i] for i in ranked_indices]
                
                # Step 4: Create hierarchical context
                context = self._create_hierarchical_context(ranked_docs)
                logger.info(f"Created context with {len(context)} characters")
                
                # Step 5: Generate answer using both context and chat history
                recent_history = history
                history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
                
                answer = await self._generate_answer(
                    query=message,
                    context=f"Previous conversation:\n{history_context}\n\nDocument context:\n{context}"
                )
            else:
                # If no documents are retrieved, fall back to regular chat
                logger.info("No documents retrieved, falling back to regular chat")
                answer = await self._regular_chat(history)
            
            # Save the answer to history
            history.append({"role": "assistant", "content": answer})
            
            # Trim history if needed
            self._trim_history(user_id)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in hierarchical PDF RAG chat: {str(e)}")
            try:
                fallback_answer = await self._regular_chat(history)
                history.append({"role": "assistant", "content": fallback_answer})
                return fallback_answer
            except:
                return "Sorry, I encountered an error and couldn't respond."
    
    def _trim_history(self, user_id: str) -> None:
        """
        Trim history to keep it within the max_history limit while preserving context.
        
        Args:
            user_id: The unique identifier for the user
        """
        history = self.user_chat_histories.get(user_id, [])
        
        if len(history) > self.max_history * 2:  # Each exchange has 2 messages
            # Keep an even number of messages to maintain user-assistant pairs
            num_to_keep = self.max_history * 2
            self.user_chat_histories[user_id] = history[-num_to_keep:]
            logger.info(f"Trimmed history for user {user_id} to {num_to_keep} messages")
    
    def reset_history(self, user_id: Optional[str] = None) -> None:
        """
        Reset chat history for a specific user or all users.
        
        Args:
            user_id: The unique identifier for the user. If None, reset all histories.
        """
        if user_id is None:
            self.user_chat_histories = {}
            logger.info("Reset all user chat histories")

        elif user_id in self.user_chat_histories:
            self.user_chat_histories[user_id] = []
            logger.info(f"Reset chat history for user {user_id}")
    
    async def add_pdf_document(self, file_path: str, doc_type: str = "general") -> bool:
        """
        Add a new PDF document to the knowledge base and update vector store.
        
        Args:
            file_path: Path to the PDF file
            doc_type: Type/category of the document
            
        Returns:
            True if document was added successfully, False otherwise
        """
        try:
            # Copy the file to the data directory if it's not already there
            filename = os.path.basename(file_path)
            destination = os.path.join(self.data_dir, filename)
            
            if file_path != destination:
                import shutil
                shutil.copy2(file_path, destination)
                file_path = destination
            
            # Load the PDF
            loader = PyPDFLoader(file_path)
            new_pdf_docs = loader.load()
            
            doc_id = str(uuid.uuid4())
            
            # Store basic metadata for the document
            doc_metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "source": file_path,
                "type": "pdf",
                "doc_type": doc_type,
                "total_pages": len(new_pdf_docs)
            }
            
            # Convert to our Document format with enhanced metadata
            new_documents = []
            for i, doc in enumerate(new_pdf_docs):
                doc_meta = dict(doc.metadata)
                doc_meta.update(doc_metadata)
                doc_meta["hierarchy_level"] = "page"
                doc_meta["hierarchy_id"] = f"{doc_id}_page_{i+1}"
                doc_meta["parent_id"] = doc_id
                
                new_documents.append(Document(
                    page_content=doc.page_content,
                    metadata=doc_meta
                ))
            
            # Add to document list
            self.documents.extend(new_documents)
            
            # Split into hierarchical chunks
            new_chunks, new_hierarchy = self._split_documents_hierarchically(new_documents)
            
            # Update hierarchy
            self.document_hierarchy.update(new_hierarchy)
            self._save_document_hierarchy()
            
            # Add chunks to existing chunks list
            self.chunks.extend(new_chunks)
            
            # Update vector store
            if self.vector_store and self.embeddings:
                # Convert to LangChain format for FAISS
                lc_docs = []
                for chunk in new_chunks:
                    lc_docs.append({
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata
                    })
                
                # Add to existing FAISS index
                new_faiss = FAISS.from_documents(lc_docs, self.embeddings)
                self.vector_store.merge_from(new_faiss)
                
                # Save the updated vector store
                self.vector_store.save_local(self.vector_store_path)
                
                logger.info(f"Added PDF document {filename} to vector store with {len(new_chunks)} chunks")
            else:
                # Initialize vector store if not already done
                self._initialize_rag()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF document: {str(e)}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all available PDF documents in the knowledge base with hierarchy info.
        
        Returns:
            List of dictionaries with document information
        """
        try:
            result = []
            
            # Use document hierarchy for more detailed information
            for doc_id, doc_info in self.document_hierarchy.items():
                filename = doc_info.get("filename", "")
                total_pages = doc_info.get("total_pages", 0)
                total_chunks = len(doc_info.get("chunks", []))
                
                # Count sections
                total_sections = 0
                for page_sections in doc_info.get("sections", {}).values():
                    total_sections += len(page_sections)
                
                # Create document summary
                doc_summary = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_pages": total_pages,
                    "total_sections": total_sections,
                    "total_chunks": total_chunks
                }
                
                result.append(doc_summary)
            
            # If no hierarchy info, fall back to basic document list
            if not result and self.documents:
                # Group documents by parent ID
                doc_groups = {}
                for doc in self.documents:
                    parent_id = doc.metadata.get("parent_id", "")
                    if parent_id not in doc_groups:
                        doc_groups[parent_id] = {
                            "doc_id": parent_id,
                            "filename": doc.metadata.get("filename", ""),
                            "pages": set()
                        }
                    doc_groups[parent_id]["pages"].add(doc.metadata.get("page", 0))
                
                # Convert to result format
                for parent_id, info in doc_groups.items():
                    result.append({
                        "doc_id": parent_id,
                        "filename": info["filename"],
                        "total_pages": len(info["pages"]),
                        "total_sections": 0,
                        "total_chunks": 0
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    async def get_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed structure of a specific document.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Dictionary with document structure information
        """
        try:
            if doc_id in self.document_hierarchy:
                return self.document_hierarchy[doc_id]
            
            # If not in hierarchy, build basic structure
            doc_structure = {
                "id": doc_id,
                "filename": "",
                "total_pages": 0,
                "sections": {},
                "chunks": []
            }
            
            # Find documents with this parent ID
            matching_docs = [doc for doc in self.documents if doc.metadata.get("parent_id") == doc_id]
            
            if matching_docs:
                doc_structure["filename"] = matching_docs[0].metadata.get("filename", "")
                doc_structure["total_pages"] = len(set(doc.metadata.get("page", 0) for doc in matching_docs))
            
            return doc_structure
            
        except Exception as e:
            logger.error(f"Error getting document structure: {str(e)}")
            return {"error": str(e)}