import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from abc_rhythm import ABCRhythmTool
import multiprocessing
import pickle
import tqdm
import json
import numpy as np
import math
from functools import partial

ORI_FOLDER = "../../all_abc"
ONSET_FILE_NAME = "../../all_onsets.pickle"

INTERLEAVED = True

GRID_RESOLUTION = 48 

tools = ABCRhythmTool(min_note_val=1/GRID_RESOLUTION)
acceptable_meters = ["4/4",'4/2',"2/2","2/4"]

def extract_onsets(abc_path, rotated=INTERLEAVED):
    try:

        onset_dic = tools.extract_unique_onsets(abc_path, rotated=rotated,activity_window=1)
        for meter in onset_dic["mpb"].values():
            if(meter not in acceptable_meters):
                return None
        return {
                "file_path":abc_path,
                "onsets":onset_dic["opvpb"],
                "mpb":onset_dic["mpb"],
                }
    except Exception as e:
        print(f"Error processing {abc_path}: {e}")
        return None


def extract_with_timeout(pool, func, arg, timeout):
    result = pool.apply_async(func, (arg,))
    try:
        return result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        print(f"Timeout on: {arg}")
        return None

def process_abc_directory(directory):
    """
    Processes all MIDI files in a directory to extract instrument information
    and saves the results to a pickle file.

    Args:
        directory (str): Path to the directory containing MIDI files.
        output_pickle (str): Path to save the extracted instrument data as a pickle file.
        midimap (pd.DataFrame): DataFrame containing MIDI program mappings.
    """
    
    # Get all MIDI files in the directory
    abc_files = [
    os.path.join(root, f) 
    for root, _, files in os.walk(directory) 
    for f in files if f.endswith('.abc')
    ]

    #abc_files = abc_files[0:10]

    if not abc_files:
        print("No ABC files found in the directory.")
        return

    #abc_files=abc_files[]
    num_processes = os.cpu_count()

        
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for f in tqdm.tqdm(abc_files, desc="Extracting Onsets"):
            res = extract_with_timeout(pool, extract_onsets, f, timeout=60)  # timeout in seconds
            results.append(res)


    # Filter out empty results
    all_onsets = [result for result in results if result]
    print(f"Onset extraction failed for {len(results)-len(all_onsets)} out of {len(results)} files")
    return all_onsets


if __name__=="__main__":
    
    
    onsets = process_abc_directory(ORI_FOLDER)
    onset_dict = {}
    print(len(onsets))
    for i in onsets:
        onset_dict[i["file_path"]]={"onsets":i["onsets"],"mpb":i["mpb"]}

    print("Successfully extracted onsets")
    
    with open(f"{ONSET_FILE_NAME}", "wb") as file:
        pickle.dump(onset_dict,file)
        

