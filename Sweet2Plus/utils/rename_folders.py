"""
Module name: rename_folders
Description: Rename folders using new string and pattern
Author: David Estrin
Date: 08-25-2024
Example usage:
    python ./rename_folders.py --input_directory /path/to/cohort/dataset/parent/folder/ --append_string Cohort1
"""
import glob, os
import argparse
import fnmatch

def append_string_to_folder(path,string,pattern='*C*_A*'):
    for root, dirs, files in os.walk(path):
        for dir_name in fnmatch.filter(dirs, pattern):
            old_path = os.path.join(root, dir_name)
            new_dir_name = f"{dir_name}_{string}"
            new_path = os.path.join(root, new_dir_name)
            os.rename(old_path, new_path)
            print(f"Renamed '{old_path}' to '{new_path}'")
    return 

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_directory',type=str,required=True)
    parser.add_argument('--append_string',type=str,required=True)
    args=parser.parse_args()
    append_string_to_folder(args.input_directory,args.append_string)