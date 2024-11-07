#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: logger.py
Description: Write and read logs to text files for logger to monitor. 
Author: David Estrin
Version: 1.0
Date: 10-15-2024
"""
import os
import time
from projectmanager.CLIlogger import Logger

# Function to write a log message to a text file
def write_log(log_dir, message):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"e_log.txt")
    with open(log_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Function to read the latest log message from a text file
def read_latest_log(log_dir):
    log_file = os.path.join(log_dir, f"e_log.txt")
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-1].strip() if lines else "No logs yet"
    except FileNotFoundError:
        return "Log file not found"
    
def update_log(log_dir, message):
    log_file = os.path.join(log_dir, f"e_log.txt")
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

if __name__ == '__main__':
    # Build an example table
    cli_log = Logger('Example Table:')
    try:
        cli_log.start_live()
        for i in range(1000):
            time.sleep(0.1)
            cli_log.update_table("124", "235", "Group24", "2024-10-13", "Processing", str(i))
    finally:
        cli_log.stop_live()

