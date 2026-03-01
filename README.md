# RE:ZEARCH
A Data-Driven Approach Leveraging extraction-based summarization as a Research Tool and Discovery of Academic Gaps



-----
## Background



Coming up with a novel research idea may be difficult, with new innovations left and right. Undergraduate students have a difficult time with coming up with ideal topics.

Both graduate and undergraduate students have problems in academic writing, organizing ideas, and writing coherently. Some find it difficult to come up with meaningful research questions for their academic works.

### Objectives



The objective of this project is to develop an NLP-powered software that facilitates the discovery of academic gaps and research trends of literature in arXiv using Extraction-Based Summarization for students, researchers, and educators to use as a research tool.

### Dataset



The dataset is a collection of 1.7 million articles from arXiv, an open-source repository of scholarly articles from various fields.

This dataset comprises article titles, authors, categories, abstracts, and the respective PDF documents.

ArXiv is an open repository of research articles operated by Cornell University.

Here is the link to the Kaggle Dataset:
https://www.kaggle.com/datasets/Cornell-University/arxiv

The dataset contains the following fields:
- id: Research paper ID
- submitter
- authors: Authors contatenated into one string
- title
- comments
- journal-ref
- doi
- report-no
- categories: Field of study
- license
- abstract
- versions
- update_date
- authors_parsed

Before we even begin the project, let us import all necessary libraries we need.

-----
## Data Preparation



First, let us gather the data. After downloading the dataset from kaggle, I placed it in a directory where all the other files related will be used.

Since the dataset will contain 1.7+ million papers, that means there will be 1.7+ million records. That simply isn't efficient if we're turning that into a DataFrame. I will be using polars, which is a library to make data analysis more efficient through speed and optimal memory use, especially for a dataset this large.

For the purposes of this project, I will only be selecting the ID (For indexing), title, categories, and abstract, as they are the only ones that fit our objective.

After the polars library does its scan, the DataFrame will be saved as a parquet file into our directory.

### Converting the Dataset to an HDF5 File (Embeddings Storage)

In this step, we convert our Parquet dataset (arxiv_data.parquet) into an HDF5 file (.h5) containing vector embeddings.

Instead of storing raw text, we will load the titles, convert them into dense vector embeddings using a transformer model, and store those vectors efficiently in an HDF5 file.

This let's us search for similarities faster, let's us use FAISS indexing, and is in general, more efficient and optimal.

The use of HDF5 will also be of help here since it supports large datasets (millions of rows). We can also load everything in chunks, making it more optimal. Lastly, compared to CSV or JSON, it is much smaller, faster, and structured for numerical arrays.

We can check first if we have a GPU available, so that we can run faster. After that, we can define our model as a SentenceTransformer. A Sentence Transformer (SBERT) is a type of transformer based-model that is used for semantic search. By creating 384-dimensional embeddings.

This is an improvement from BERT's limitations. Both are used for semantic analysis, but SBERT analyzes text in sentence level, has a fixed-size embedding, and is best for semantic search, which is useful for our project.

```
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embedding_size = 384
```



The code below will initialize the parquet DataFrame, then the model. After that, we will begin embedding the H5PY file. We will create a file and allocate space for this. Then, for every chunk with a size of 100,000, we will slowly write the embeddings per 256 samples.

After the iterations, the file will be saved to the embeddings path.

### Building the FAISS Index from HDF5 Embeddings

This block of code builds a FAISS similarity search index from the embeddings stored in the HDF5 file. After previously converting all titles into 384-dimensional numerical vectors and saving them efficiently, we will then transform those stored embeddings into a structure that enables fast semantic search.

Instead of scanning every vector manually during a query, FAISS organizes the vectors in a way that allows efficient nearest-neighbor retrieval using Euclidean distance. The embeddings are loaded from the HDF5 file in chunks to prevent excessive memory usage. Each chunk of vectors is added to the index incrementally until all embeddings are indexed.

Finally, the completed FAISS index is saved to disk as a .index file, which can later be loaded to perform similarity searches over the entire dataset.

-----
## Model Creation



In this next step, we will begin the creation of the system itself.

The first function, read_parquet_rows, is designed to efficiently retrieve specific rows from the parquet file without loading the entire dataset into memory. Since the FAISS index only returns vector positions (row indices), this function maps those indices back to the original dataset.

It determines which row group a requested index belongs to, reads only from that row group, and extracts the exact row needed. This lookup approach keeps memory usage low.

The second function, rezearch_query_safe, performs the actual semantic search. It takes a proposal title as input and converts it into a vector embedding using the same transformer model that was used to embed the dataset.

It then queries the FAISS index to retrieve the top k (5 by default) most similar embeddings based on L2 distance. The returned indices correspond to rows in the original dataset, so the function calls read_parquet_rows to fetch the titles, categories, and abstracts of the matching papers.

It also returns a “gap score,” which represents the distance between the query and the closest match—this can be interpreted as how semantically similar the proposal is to existing research.

The third block uses LexRank for extractive summarization. The summarize_and_extract_keywords function takes a research abstract and generates a concise summary. It doesn't generate text, rather identifies the most central sentences based on graph-based ranking, making it computationally efficient and suitable for research abstracts.

After that, it uses KeyBERT, using  transformer embeddings to identify the most semantically meaningful keywords and keyphrases in a document. This returns a  condensed explanation of the research and its main terms, making it easier to understand the focus of the retrieved paper.

Lastly, we can proceed to building them all together.

The function research_dashboard takes in a proposal title and uses the previous function to return top research papers, their summarized abstract, and the relevant keywords.

To better visualize the gap score, the function visualize_top_papers takes in the prompt used and looks into the top indices to see the relevant rows closest to it, mapping the top similar maps in a scallter plot.

### Interactive System



Now that everything is all set, we can then try to implement a way to allow users to test the system, but for now, we will use a notebook. Let us first instantiate all the necessary variables.

Next, we can define the code, creating a widget in our notebook. Essentially, this will create a textbox where we can enter our prompt, and when we click the button, the system will run, taking the top matching papers to our prompt, and return those papers as well as the other necessary information.

-----
## Conclusion



In this notebook, we developed a complete semantic research retrieval system powered by transformer embeddings and FAISS similarity search. Starting from raw arXiv metadata, we converted textual data into dense vector representations, stored them efficiently in HDF5 format, and indexed them for fast nearest-neighbor search. We then implemented a retrieval pipeline that:

- Embeds a user-provided research proposal
- Searches for the most semantically similar papers
- Retrieves relevant metadata from the dataset
- Summarizes abstracts for quick interpretation
- Extracts key thematic keywords
- Displays results interactively using a notebook widget

This project can provide assistance for:
- Literature review automation
- Proposal novelty checking
- Semantic document retrieval
- Academic recommendation systems

-----
## Recommendation



There are still a lot to improve upon this first version of ReZearch. We can look into switching to other indexing approaches other than IndexFlatL2 such as IndexIVFFlat or HNSW, but those require better CPU and memory. We can also add cosine similarity, since gap distance also uses L2.

In terms of the semantic search itself, we can add a hybrid search, combining keywords and semantics, as well as adding a scoring system to determine accuracy. Research integrity can also be an addition, where we can integrate rankings of the research papers retrieved based on their peer review status.

Lastly, the system itself is very memory intensive, but implementing a web application is not out of the table.

Overall,

This project served as a mini mini thesis to one of our initial title proposals that we didn't choose to continue. It combines contemporary NLP techniques with efficient vector indexing to build a semantic search systems.

The pipeline is modular and extensible, though resource extensive. The original dataset has been updated from an initial 1.7 million to a whopping 2.9 million. If the system were to be updated, it would require an estimated 1-2 hours, 10 gb of storage, and a better memory and cpu.
