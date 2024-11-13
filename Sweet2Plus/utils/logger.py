#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: logger.py
Description: Write and read logs to text files for logger to monitor. 
Author: David Estrin
Version: 1.0
Date: 10-15-2024
"""
import os, glob
import time
from projectmanager.CLIlogger import Logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import ipdb
import argparse

""" Custom Functions
Meant for writing, reading and updating 'e_log.txt' files.
"""
def write_log(log_dir, cage, mouse, group, day, message):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"e_log_C{cage}_M{mouse}_G{group}_day{day}.txt")
    with open(log_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Function to read the latest log message from a text file
def read_latest_log(log_file):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else "No logs yet"
    except FileNotFoundError:
        return "Log file not found"
    
def update_log(log_dir, cage, mouse, group, day, message):
    log_file = os.path.join(log_dir, f"e_log_C{cage}_M{mouse}_G{group}_day{day}.txt")
    try:
        with open(log_file, "r+") as f:
            lines = f.readlines()
            if lines:
                # Overwrite the last line with the new message
                lines[-1] = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
            else:
                # If file is empty, write the message as the first line
                lines.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            f.seek(0)  # Go back to the start of the file
            f.writelines(lines)  # Write the updated content
            f.truncate()  # Remove any leftover lines if file was longer
    except FileNotFoundError:
        # If the file doesn't exist, write the new message as the first line
        write_log(log_dir, message)

""" Custom logger 
Written for s2p pipeline which utalizes files for logging processes
"""
class s2p_logger(Logger):
    def __init__(self, table_name, log_directory):
        super().__init__(table_name)
        self.log_directory = log_directory

    def collect_logs(self):
        def parse_log(filename,log_oh):
            # parse filename
            _,filename=file.split('e_log_')
            filename,_=filename.split('.t')
            cage,subject,group,day=filename.split('_')

            # parse log info
            _,_,_,message=log_oh.split('-')
            step,progress=message.split('%')

            return cage, subject, group, day, step, progress

        # Find all the e-file logs
        e_logs=glob.glob(os.path.join(self.log_directory,'*e_log*.txt*'))

        # open each file up and grab relevant data
        for file in e_logs:
            log_oh = read_latest_log(file)

            # parse log_oh
            parsed_info=parse_log(filename=file,log_oh=log_oh)

            # Update table with parsed log info
            self.update_table(cage=parsed_info[0], subject=parsed_info[1], group=parsed_info[2], day=parsed_info[3], step=parsed_info[4], progress=parsed_info[5])

class eloghandler(FileSystemEventHandler):
    def __init__(self,mylogger):
        super().__init__()
        self.mylogger=mylogger

    def on_modified(self, event):
        if event.src_path.endswith(".txt"):
            self.mylogger.collect_logs()

    def on_created(self, event):
        if event.src_path.endswith(".txt"):
            self.mylogger.collect_logs()

    def on_deleted(self, event):
        if event.src_path.endswith(".txt"):
            print(f"File deleted: {event.src_path}")

def delete_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def watch_directory(table_name='Example Table:',directory=r".\logs",test_mode=False,force_delete_logs=False):
    mylogger=s2p_logger(table_name,directory)
    mylogger.start_live()

    event_handler = eloghandler(mylogger)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    keep_running=True

    if force_delete_logs:
        delete_all_files(directory)

    try:
        while keep_running:
            time.sleep(1)  # Keep the script running

            # Create files if code is set to test mode
            if test_mode:
                delete_all_files(directory)
                write_log(directory, '355','2','CORT', '14','Initializing Sweet 2 P%0')
                write_log(directory, '356','2', 'CORT', '14','Initializing Sweet 2 P%0')
                write_log(directory, '354','2','CORT', '14','Initializing Sweet 2 P%0')
                write_log(directory, '35123','2','CONTROL','14','Initializing Sweet 2 P%0')
                write_log(directory, '3','2','CONTROL','14','Initializing Sweet 2 P%0')
                time.sleep(5)
                update_log(directory, '355','2','CORT','14','Running MLP on F data%0')
                update_log(directory, '356','2','CORT','14','Running MLP on F data%0')
                update_log(directory, '354','2','CORT','14','Running MLP on F data%0')
                update_log(directory, '35123','2','CONTROL','14','Running MLP on F data%0')
                update_log(directory, '3','2','CONTROL','14','Running MLP on F data%0')
                keep_running=False
                observer.stop()
                mylogger.stop_live()

    except KeyboardInterrupt:
        observer.stop()
        mylogger.stop_live()

    observer.join()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--log_folder',required=True,type=str)
    args=parser.parse_args()
    watch_directory(table_name='EstrinJohnson_2P_Experiment',directory=args.log_folder,force_delete_logs=True)


