import os
from ingestion import ingest_text, ingest_pdf

def get_file_input() -> str:
    """Prompt user for file path and validate it."""
    while True:
        file_path = input("Enter the path to your file: ").strip()

        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            continue

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.txt', '.pdf']:
            print(f"Error: Unsupported file type. Only .txt and .pdf files are supported.")
            continue

        return file_path

def get_chunk_params() -> tuple[int, int]:
    """Prompt user for chunk size and overlap."""
    while True:
        try:
            chunk_size = int(input("Enter chunk size (default 1000): ") or "1000")
            if chunk_size <= 0:
                print("Chunk size must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            chunk_overlap = int(input("Enter chunk overlap (default 0): ") or "0")
            if chunk_overlap < 0:
                print("Chunk overlap cannot be negative.")
                continue
            if chunk_overlap >= chunk_size:
                print("Chunk overlap must be less than chunk size.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    return chunk_size, chunk_overlap

def main():
    print("=== Document Ingestion Tool ===")
    print("This tool will help you ingest text and PDF files into the vector database.")
    print("Supported file types: .txt, .pdf\n")

    try:
        # Get and validate file input
        file_path = get_file_input()

        # Get chunk parameters
        chunk_size, chunk_overlap = get_chunk_params()

        print("\nStarting ingestion process...")
        print(f"File: {file_path}")
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {chunk_overlap}\n")

        # Determine ingestion method based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.txt':
            ingest_text(file_path, chunk_size, chunk_overlap)
        elif file_ext == '.pdf':
            ingest_pdf(file_path, chunk_size, chunk_overlap)

        print(f"\n✅ Successfully ingested: {file_path}")

    except KeyboardInterrupt:
        print("\n\nIngestion cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")

if __name__ == "__main__":
    main()
