import os

def read_files(directory):
    file_data = []
    exclude_dirs = {'node_modules', 'dist', 'build', '.git'}
    exclude_files = {'package-lock.json', 'yarn.lock', 'README.md', 'LICENSE'}

    for root, dirs, files in os.walk(directory):
        # Skip the excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file in exclude_files or file.startswith('.'):
                continue  # Skip the excluded files
            if file.endswith(('.jsx', '.js', '.scss', '.css', '.html')):  # Include project(react) files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data.append(f.read())
                except UnicodeDecodeError as e:
                    print(f"Could not read file {file_path} due to a UnicodeDecodeError: {e}")
    return "\n".join(file_data)  # Combine file contents into a single string

# Directory to read
codebase_data = read_files('C:/Users/HP/Documents/work/portfolio_nick')

# Write the combined data to a single file
with open('combined_codebase.txt', 'w', encoding='utf-8') as f:
    f.write(codebase_data)

# Print out the collected data for verification
# for filename, content in codebase.items():
#     print(f"Filename: {filename}\nContent:\n{content[:100]}...\n")