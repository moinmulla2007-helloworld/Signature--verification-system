import os
import zipfile
import shutil

def setup_cedar_dataset(zip_filename="archive.zip", target_dir="dataset"):
    # Change "archive.zip" above to exactly match the name of the file you downloaded
    
    genuine_dir = os.path.join(target_dir, 'genuine')
    forged_dir = os.path.join(target_dir, 'forged')
    
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    if not os.path.exists(zip_filename):
        print(f"Error: {zip_filename} not found in the current folder!")
        return

    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall("temp_cedar")
        
    print("Sorting images into genuine and forged folders...")
    
    for root, dirs, files in os.walk("temp_cedar"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Sort based on folder or file names containing 'forg'
                if 'forg' in file.lower() or 'forg' in root.lower():
                    shutil.move(file_path, os.path.join(forged_dir, file))
                else:
                    shutil.move(file_path, os.path.join(genuine_dir, file))
                    
    shutil.rmtree("temp_cedar")
    print(f"Success! Real signature dataset loaded into '{target_dir}'.")

if __name__ == "__main__":
    setup_cedar_dataset()