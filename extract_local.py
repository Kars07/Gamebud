import tarfile
import json
import pandas as pd
import os

# The file you just downloaded
TAR_PATH = "SHARD_0000.tar.gz"

print(f"Opening {TAR_PATH}...")
with tarfile.open(TAR_PATH, "r:gz") as tar:
    # 1. Search for the first valid parquet file
    target_member = None
    target_folder = None
    
    for member in tar:
        if "actions_raw.parquet" in member.name:
            target_member = member
            # Get the folder name (Video ID)
            target_folder = os.path.dirname(member.name)
            break
    
    if not target_member:
        print("Error: Could not find any parquet files in the archive.")
        exit()

    print(f"Found Sample: {target_folder}")

    # 2. Extract the Action Data
    f = tar.extractfile(target_member)
    df = pd.read_parquet(f)
    df.to_parquet("training_actions.parquet")
    print("   -> Saved 'training_actions.parquet'")

    # 3. Find and Extract the Metadata (Video URL)
    # We look for the metadata.json in the SAME folder
    meta_name = f"{target_folder}/metadata.json"
    try:
        meta_member = tar.getmember(meta_name)
        f_meta = tar.extractfile(meta_member)
        meta = json.load(f_meta)
        
        with open("training_metadata.json", "w") as out:
            json.dump(meta, out, indent=2)
        print("   -> Saved 'training_metadata.json'")
        
        print("\n" + "="*40)
        print("READY FOR NEXT STEP")
        print("="*40)
        print(f"YouTube URL: {meta['original_video']['url']}")
        print(f"Timestamps: {meta['original_video']['start_time']}s - {meta['original_video']['end_time']}s")
        print(f"Joystick Data Shape: {df[['j_left', 'j_right']].head(1).values}")

    except KeyError:
        print(f"Could not find metadata file: {meta_name}")