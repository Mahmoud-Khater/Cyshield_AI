# Semantic Search and Keyword Extraction

## Introduction

This project addresses the challenge of searching through academic documents and extracting meaningful keywords from atricles. Traditional keyword-based search methods often fail to capture semantic relationships between concepts, while pure text matching may miss relevant content due to vocabulary variations.

The goal of this project is to develop a hybrid semantic search system that combines TF-IDF vectorization with modern NLP techniques to:
- Enable natural language querying of document collections
- Extract hot keywords both globally across all documents and document-specifically
- Provide ranked search results with source attribution and metadata preservation
- Demonstrate the effectiveness of combining traditional IR methods with modern vector search approaches

## Data Description

### Dataset Overview
- **Document Collection**: 2 PDF files (research papers)
- **File Sources**: 
  - `2025.cl-1.1.pdf`
  - `2025.cl-2.5.pdf`
- **Total Pages**: 47 pages across both documents
- **Text Chunks**: 289 chunks after recursive text splitting
- **Processing Method**: No training/testing split - search and keyword extraction

### Data Processing Pipeline
- **Text Extraction**: PDF content extracted using PyPDFLoader
- **Chunking Strategy**: Recursive character text splitter with 500-character chunks and 50-character overlap
- **Vectorization**: TF-IDF with English stopword removal
- **Feature Space**: Variable vocabulary size based on document content after stopword filtering
- **Vector Store**: FAISS index for efficient similarity search

### Data Characteristics
| Metric | Value |
|--------|-------|
| Total Documents | 2 PDFs |
| Total Pages | 47 pages |
| Text Chunks | 289 chunks |
| Chunk Size | 500 characters |
| Chunk Overlap | 50 characters |
| Vectorization Method | TF-IDF |
| Search Index | FAISS |

*Table 1: Dataset and processing statistics for the semantic search system*

## Baseline Experiments

### Experiment Goal
The baseline experiment aims to establish a functional semantic search system using TF-IDF vectorization combined with FAISS vector store, and evaluate the system's ability to retrieve relevant content for natural language queries about computational linguistics topics.

### Experimental Setup
- **Search Method**: TF-IDF + FAISS hybrid approach
- **Query Processing**: Natural language queries converted to TF-IDF vectors
- **Retrieval**: Top-3 most similar chunks returned per query
- **Keyword Extraction**: Global and document-specific hot keywords using TF-IDF scores

### Test Queries and Results
Two representative queries were tested to evaluate system performance:

**Query 1**: "What is dotless arabic?"
- **Top Result**: Page 27 from 2025.cl-2.5.pdf discussing dotless Arabic text representation
- **Content Relevance**: High - directly addresses the query topic
- **Source Attribution**: Correct page and document identification

**Query 2**: "define Computational Linguistics" 
- **Top Results**: Multiple references from 2025.cl-1.1.pdf including journal citations
- **Content Relevance**: Moderate - found bibliographic references rather than definitions
- **Source Attribution**: Correct page and document identification

### Global Keyword Extraction Results
| Rank | Keyword | Relevance |
|------|---------|-----------|
| 1 | text | High |
| 2 | dotless | High |
| 3 | arabic | High |
| 4 | dotted | High |
| 5 | language | Medium |

*Table 2: Top-5 global hot keywords extracted from the document collection*

### Baseline Conclusions
The baseline TF-IDF + FAISS approach successfully demonstrates:
- **Functional semantic search** with relevant result retrieval
- **Effective keyword extraction** identifying domain-specific terms
- **Proper source attribution** maintaining document and page metadata
- **Reasonable performance** for small document collections

However, the system shows limitations in handling definitional queries where content may be referenced rather than explicitly defined.

## Other Experiments

### Experiment 1: Chunk Size Optimization

**Goal**: Evaluate the impact of different chunk sizes on search result quality and keyword extraction effectiveness.

**Steps**:
1. Tested chunk sizes: 300, 500, and 700 characters
2. Maintained 50-character overlap across all configurations
3. Evaluated search result relevance for the same test queries
4. Analyzed keyword extraction consistency

**Results**:
- **300 characters**: More granular results but fragmented context
- **500 characters**: Optimal balance between granularity and context preservation
- **700 characters**: Better context but reduced precision for specific topics

**Conclusion**: 500-character chunks provide the best balance for simple questions in academic documents, offering sufficient context while maintaining search precision.

### Experiment 2: Document-Specific vs Global Keyword Analysis

**Goal**: Compare the effectiveness of global keyword extraction versus document-specific keyword identification for understanding content themes.

**Steps**:
1. Extract global keywords across entire document collection
2. Extract document-specific keywords for individual papers
3. Analyze keyword overlap and uniqueness
4. Evaluate relevance to document topics

**Results**:
- **Global Keywords**: ['text', 'dotless', 'arabic', 'dotted', 'language']
- **Document-Specific**: More focused terms like 'tokenization', 'vocabulary', 'dataset'
- **Overlap**: Core domain terms appear in both analyses
- **Uniqueness**: Document-specific extraction reveals specialized topics

**Conclusion**: Document-specific keyword extraction provides more targeted insights for individual papers, while global extraction identifies overarching themes across the collection.

## Overall Conclusion

This project successfully demonstrates the implementation of a hybrid semantic search system for academic document analysis. The combination of TF-IDF vectorization with FAISS indexing provides an effective approach for small to medium-sized document collections in specialized domains like computational linguistics.

**Key Achievements**:
- Functional semantic search with natural language query support
- Effective keyword extraction at both global and document levels
- Proper integration of traditional IR methods with modern vector search
- Configurable parameters allowing adaptation to different document types

**System Limitation**:
- Limited evaluation methodology without ground truth data

**Practical Applications**:
The system demonstrates potential for research paper analysis, literature review assistance, and academic content discovery in specialized domains.

## Tools and Technologies Used

### Programming Languages and Frameworks
- **LangChain**: Document processing and vector store management
- **Scikit-learn**: TF-IDF vectorization and machine learning utilities
- **FAISS**: Efficient similarity search and clustering
- **NumPy**: Numerical computations and array operations

### Libraries and Dependencies
- **PyPDF**: PDF document parsing and text extraction
- **Google AI Generative Language**: Gemini model integration
- **OpenAI**: Embedding model services
- **LangChain Community**: Extended document loaders and processors
- **LangChain OpenAI**: OpenAI service integrations

### Development Environment
- **Google Colab**: Cloud-based Jupyter notebook environment
- **Jupyter Notebook**: Interactive development and experimentation
- **Google Drive**: Document storage and access

### API Services
- **Google AI API**: Access to Gemini 2.5 Flash language model
- **OpenAI API**: Text embedding services (text-embedding-3-large)

## External Resources

### Documentation and References
- **LangChain Documentation**: Primary reference for framework implementation and best practices
- **FAISS Documentation**: Vector similarity search implementation guidance
- **Scikit-learn Documentation**: TF-IDF vectorization and preprocessing techniques
- **Google AI Documentation**: Gemini model integration and API usage

## Project Reflection

### 1. What was the biggest challenge you faced when carrying out this project?

The biggest challenge was integrating multiple different frameworks and APIs while maintaining compatibility between LangChain's document processing pipeline and scikit-learn's TF-IDF implementation. Creating a custom embeddings wrapper to bridge the gap between traditional TF-IDF vectorization and LangChain's modern vector store interface required careful attention to data format conversions and ensuring consistent behavior across the different components. Additionally, managing API key authentication across multiple services (Google AI and OpenAI) in the Colab environment while maintaining functionality presented configuration challenges.

### 2. What do you think you have learned from the project?

This project provided valuable hands-on experience with hybrid search architectures that combine traditional information retrieval methods with modern NLP techniques. I learned how to effectively integrate multiple frameworks (LangChain, scikit-learn, FAISS) to create a cohesive system, and gained practical experience with vector stores and semantic search implementation. The project also highlighted the importance of proper text chunking strategies for maintaining context while enabling granular search capabilities. Additionally, working with real academic documents demonstrated the challenges of processing diverse document formats and extracting meaningful insights from specialized domain content. The experience reinforced the value of modular design patterns that allow for easy component replacement and system scaling.
