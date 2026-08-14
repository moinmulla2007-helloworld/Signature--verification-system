import os
import shutil

# The folders you want to combine
dirs_to_merge = ["dataset", "Custom_Dataset"]
combined_dir = "Combined_Dataset"

# Create the new master folders
os.makedirs(os.path.join(combined_dir, "genuine"), exist_ok=True)
os.makedirs(os.path.join(combined_dir, "forged"), exist_ok=True)

print(f"Merging files into {combined_dir}...")

for source_dir in dirs_to_merge:
    if not os.path.exists(source_dir):
        print(f"Skipping {source_dir} (Folder not found).")
        continue
        
    for category in ["genuine", "forged"]:
        source_cat_dir = os.path.join(source_dir, category)
        if not os.path.exists(source_cat_dir):
            continue
            
        for filename in os.listdir(source_cat_dir):
            src_path = os.path.join(source_cat_dir, filename)
            
            # Prefix the filename with the original folder name 
            # so 'dataset/gen_01.png' doesn't overwrite 'Custom_Dataset/gen_01.png'
            new_filename = f"{source_dir}_{filename}"
            dest_path = os.path.join(combined_dir, category, new_filename)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)

print("Datasets successfully combined!")