
import pathlib
import shutil
import re
from pathlib import Path

def move_file(fpath, limit, move_to_folder): # '../data/small_files'
    fsize = Path(fpath).stat().st_size
    fsize_mb = fsize/float(1<<20)
    
    if fsize_mb < limit:
        create_folder(move_to_folder)
        shutil.move(fpath, move_to_folder+fpath.split('\\')[-1])
        print('Moved file: ', fpath, ' to folder ', move_to_folder)


def create_folder(folder_path):
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)


def clean_excessive_spaces(_str, remove_all= False):
    _str = re.sub('\n+', '\n', _str)
    _str = re.sub('\t+', '\t', _str)
    
    if remove_all:
        _str = re.sub('\n', '', _str)
        _str = re.sub('\t', ' ', _str)
    return _str