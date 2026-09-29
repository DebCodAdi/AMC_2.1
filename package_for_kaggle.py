"""Package the NTRO Signal Analysis codebase for Kaggle training (excludes large models and logs)."""
import os
import zipfile

def package_codebase():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_zip = os.path.join(root_dir, "kaggle_code.zip")

    # Folders and files to exclude from upload zip
    exclude_dir_names = {"__pycache__", ".pytest_cache", ".git", ".venv"}
    exclude_exts = {".ckpt", ".log", ".pyc", ".pkl", ".tar.bz2", ".zip"}
    top_models_dir = os.path.abspath(os.path.join(root_dir, "models"))

    print(f"[*] Packaging codebase from: {root_dir}")
    count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dir_names and os.path.abspath(os.path.join(root, d)) != top_models_dir]
            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, rel_path)
                count += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[+] Successfully created: {output_zip}")
    print(f"[+] Packaged {count} files ({size_mb:.2f} MB). Ready to upload to Kaggle!")

if __name__ == "__main__":
    package_codebase()
