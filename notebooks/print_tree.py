import os

def generate_tree(startpath, output_file="directory_structure.txt"):
    # Folders we want to completely ignore to keep the tree clean
    ignore_dirs = {'venv', '.git', '__pycache__', '.ipynb_checkpoints', '.vscode'}
    
    # Extensions we want to hide so we don't print 8,000 image files
    ignore_extensions = {'.npy', '.tif', '.jpg', '.png'}

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("🌳 WILDFIRE PREDICTION SYSTEM - DIRECTORY STRUCTURE\n")
        f.write("=" * 60 + "\n\n")

        for root, dirs, files in os.walk(startpath):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # Calculate depth for indentation
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            
            folder_name = os.path.basename(root)
            if level == 0:
                folder_name = "wildfire-prediction-system (Root)"
                
            f.write(f"{indent}📁 {folder_name}/\n")
            
            subindent = ' ' * 4 * (level + 1)
            
            # Filter out mass data files, but count them
            visible_files = []
            hidden_count = 0
            
            for file in files:
                if any(file.endswith(ext) for ext in ignore_extensions):
                    hidden_count += 1
                else:
                    visible_files.append(file)
            
            # Print visible files (like .py, .csv, .ipynb)
            for file in visible_files:
                f.write(f"{subindent}📄 {file}\n")
                
            # Print summary of hidden files if any exist
            if hidden_count > 0:
                f.write(f"{subindent}📦 [... {hidden_count} data/image files ...]\n")

if __name__ == "__main__":
    current_dir = os.getcwd()
    generate_tree(current_dir)
    print("✅ Directory tree successfully saved to 'directory_structure.txt'")