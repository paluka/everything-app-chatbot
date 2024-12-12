import os

def load_files(repo_dir):
    """Load text files from a directory and return their contents as a list."""
    
    exclude_dirs = ["node_modules", ".next"]  # Default directories to exclude
    include_extensions = [".js", ".jsx", ".ts", ".tsx", ".json", ".schema"] 
    documents = []

    # Walk through all files in the repository directory
    for root, dirs, files in os.walk(repo_dir):
        # Exclude specific directories by modifying the dirs list in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in include_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    # documents.append(f.read())
                    # code = f.read()
                    # Prepend the file name or context to the code
                    # documents.append(f"### File: {file_path}\n{code}")
                    documents.append(f"File: {file_path}")

    print(f"\nFound {len(documents)} in {repo_dir}")
    return documents
