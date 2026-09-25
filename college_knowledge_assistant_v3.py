import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

client=genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
COLLEGE_DATA=Path("college_data")

#1.LOAD COLLEGE DOCUMENTS

def load_documents():

    documents = []

    for file in COLLEGE_DATA.glob("*.pdf"):

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append({
            "name": file.name,
            "text": text
        })

    return documents

#2.CREATE CHUNKS
def create_chunks(text,chunk_size=500):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),chunk_size):
        chunk="".join(
            words[i:i + chunk_size]
          
        )
        chunks.append(chunk)
        return chunks  
 #3.create embeddings
def create_embedding(text):
    response=client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )      
    return response.embeddings[0].values

#main
def main():
    print("=" * 60)
    print("       COLLEGE KNOWLEDGE ASSISTANT V3")
    print("=" * 60)

    documents = load_documents()

    if not documents:

        print("\nNo PDF documents found.")

        return

    print(
        f"\nLoaded {len(documents)} college document(s)."
    )

    all_chunks = []

    # Create chunks

    for document in documents:

        chunks = create_chunks(
            document["text"]
        )

        for number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append({
                "name": document["name"],
                "chunk_number": number,
                "text": chunk
            })

    print(
        f"Created {len(all_chunks)} chunks."
    )

    # Create embeddings

    print("\nCreating embeddings...\n")

    for chunk in all_chunks:

        chunk["embedding"] = create_embedding(
            chunk["text"]
        )

        print(
            f"Embedded: "
            f"{chunk['name']} | "
            f"Chunk {chunk['chunk_number']}"
        )

    # Show first embedding

    print("\n" + "=" * 60)

    print("FIRST CHUNK")

    print("=" * 60)

    print(
        all_chunks[0]["text"]
    )

    print("\n" + "=" * 60)

    print("FIRST EMBEDDING")

    print("=" * 60)

    print(
        all_chunks[0]["embedding"]
    )

    print(
        "\nEmbedding length:",
        len(all_chunks[0]["embedding"])
    )


if __name__ == "__main__":
    main()
