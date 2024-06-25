
# Verschiedene Loader für verschiedene Dateitypen


```python

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': PyMuPDFLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
}

# Define a function to create a DirectoryLoader for a specific file type
def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
    )

# Create DirectoryLoader instances for each file type
pdf_loader = create_directory_loader('.pdf', '/path/to/your/directory')
xml_loader = create_directory_loader('.xml', '/path/to/your/directory')
csv_loader = create_directory_loader('.csv', '/path/to/your/directory')

# Load the files
pdf_documents = pdf_loader.load()
xml_documents = xml_loader.load()
csv_documents = csv_loader.load()
```

Quelle: [Github](https://github.com/langchain-ai/langchain/discussions/18559)
