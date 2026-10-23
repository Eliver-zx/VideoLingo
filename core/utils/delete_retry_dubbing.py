import os
import shutil

from core.utils.models import resolve_task_path

def delete_dubbing_files():
    files_to_delete = [
        resolve_task_path(os.path.join("output", "dub.wav")),
        resolve_task_path(os.path.join("output", "dub.mp3")),
        resolve_task_path(os.path.join("output", "audio", "dub.wav")),
        resolve_task_path(os.path.join("output", "normalized_dub.wav")),
        resolve_task_path(os.path.join("output", "dub.srt")),
        resolve_task_path(os.path.join("output", "dub_src.srt")),
        resolve_task_path(os.path.join("output", "output_dub.mp4")),
        resolve_task_path(os.path.join("output", "audio", "tts_tasks.xlsx")),
        resolve_task_path(os.path.join("output", "log", "dub_truncated.json")),
        resolve_task_path(os.path.join("output", ".dubbing_done")),
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    for folder_name in ("segs", "tmp"):
        folder_path = resolve_task_path(os.path.join("output", "audio", folder_name))
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder and contents: {folder_path}")
            except Exception as e:
                print(f"Error deleting folder {folder_path}: {str(e)}")
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    delete_dubbing_files()