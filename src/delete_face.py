import json
import shutil
import time
from pathlib import Path
import numpy as np

try:
    from src import config
except ImportError:
    # Fallback if run directly instead of as module
    import config

def main():
    db_path = config.DB_NPZ_PATH
    json_path = Path("data/db/face_db.json")
    crops_dir = Path("data/enroll")

    if not db_path.exists():
        print("No face database found at data/db/face_db.npz.")
        return

    # Load database
    try:
        data = np.load(str(db_path), allow_pickle=True)
        db = {k: data[k] for k in data.files}
    except Exception as e:
        print(f"Error loading face database: {e}")
        return

    if not db:
        print("Database is empty. No faces to delete.")
        return

    names = sorted(db.keys())
    print("\nEnrolled Faces:")
    for i, name in enumerate(names, 1):
        print(f" {i}. {name}")

    print("\nEnter the number or name of the face to delete (or press Enter to cancel): ", end="")
    choice = input().strip()
    if not choice:
        print("Canceled.")
        return

    target_name = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            target_name = names[idx]
    elif choice in db:
        target_name = choice

    if not target_name:
        print(f"Invalid selection: '{choice}'.")
        return

    print(f"\nAre you sure you want to delete '{target_name}'? (y/N): ", end="")
    confirm = input().strip().lower()
    if confirm not in ['y', 'yes']:
        print("Deletion canceled.")
        return

    # 1. Remove from database dictionary
    del db[target_name]
    
    # Save npz database
    try:
        np.savez(str(db_path), **db)
        print(f"Successfully removed '{target_name}' from {db_path.name}.")
    except Exception as e:
        print(f"Error saving npz database: {e}")
        return

    # 2. Update metadata json
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            meta["names"] = sorted(db.keys())
            meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            print(f"Updated metadata in {json_path.name}.")
        except Exception as e:
            print(f"Warning: could not update metadata json: {e}")

    # 3. Clean up the enroll images folder
    person_crops_dir = crops_dir / target_name
    if person_crops_dir.exists() and person_crops_dir.is_dir():
        try:
            shutil.rmtree(person_crops_dir)
            print(f"Deleted crops directory: {person_crops_dir.resolve().relative_to(Path.cwd())}")
        except Exception as e:
            print(f"Warning: could not delete crops directory: {e}")
            
    print("\nDeletion complete!")

if __name__ == "__main__":
    main()
