# In main.py

from src.data_loader import load_all_pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ... other imports

# 1. Load all documents from the 'data' folder
all_policy_docs = load_all_pdfs(data_folder_path="data")

# 2. Split all the loaded documents into chunks
# The text_splitter works on the list of all docs just like it did for one.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_policy_docs)

print(f"\nTotal chunks created from all documents: {len(splits)}")

# 3. From here, the process is the same:
#    - Create the vector store from these `splits`.
#    - Define your retriever and chain.
#    - Ask your questions!
# ...
# Assuming 'all_policy_docs' is the variable holding your loaded data
all_policy_docs = load_all_pdfs(data_folder_path="data")

# --- Let's inspect the data ---

# 1. See the total number of pages loaded from all PDFs
print(f"\n📄 Total pages loaded: {len(all_policy_docs)}")

# 2. See the content of the VERY FIRST page
for i in range(len(all_policy_docs)):
    print(f"\n--- Content of {i + 1}th Page ---")
    print(all_policy_docs[i].page_content)
    print("--------------------------")

# 3. See the metadata for the first page
# Metadata tells you which file it came from and the page number.
print("\n🔍 Metadata of First Page:")
print(all_policy_docs[0].metadata)
print("--------------------------")