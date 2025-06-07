import os
import sys
import multiprocessing
import pickle
import tqdm
import json
import numpy as np
import math
from functools import partial
import pyinmean
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from abc_rhythm import ABCRhythmTool
import pandas as pd


ONSET_FILE_NAME = "../../all_onsets.pickle"
OUTPUT_FILE_NAME = "../../labeled_onsets.csv"

INTERLEAVED = True

GRID_RESOLUTION = 48 

tools = ABCRhythmTool(min_note_val=1/GRID_RESOLUTION)


def label_onset(entry, rotated=INTERLEAVED):
    for voice in entry["onsets"].keys():
        onset_pool = []
        onsets_per_bar = entry["onsets"][voice]
        for bar in onsets_per_bar:
            onset_pool+=bar
        onset_pool = [int(onset) for onset in onset_pool]
        onset_pool.sort()

        baroffsets = []
        for i in range(len(entry["mpb"])):
            meter = entry["mpb"][i]
            beats, beat_unit = map(int, meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4
            tmp_resolution = beat_unit if GRID_RESOLUTION==None else GRID_RESOLUTION
            bar_duration = int((beats / beat_unit) * tmp_resolution)
            baroffsets.append(bar_duration)
        data = {}
        if(len(onset_pool)>3):
            voice_spect = pyinmean.get_normalized_ima(onset_pool,True)
            voice_met = pyinmean.get_normalized_ima(onset_pool,False)
            voice_met = tools.sc.extend_metric_weight(voice_met,onset_pool)
            ima_s_arr = tools.arrange_ima(voice_spect,entry["mpb"])
            ima_m_arr = tools.arrange_ima(voice_met,entry["mpb"])
            sync = tools.calculate_syncopation_score(onsets_per_bar,baroffsets,entry["mpb"])
            distances = tools.calculate_avg_onset_distance(onsets_per_bar,entry["mpb"])
            data[voice] = {"metric_arranged":ima_m_arr,"spectral_arranged":ima_s_arr,"syncopation":sync,"distances":distances}
    
    return data

def extract_with_timeout(pool, func, arg, timeout):
    result = pool.apply_async(func, (arg,))
    try:
        return result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        print(f"Timeout on: {arg}")
        return None

def process_abc_dict(onset_data):
    
    num_processes = os.cpu_count()

    failed=0
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = {}
        for key,entry in tqdm.tqdm(onset_data.items(), desc="Extracting Onsets"):
            res = extract_with_timeout(pool, label_onset, entry, timeout=60)  # timeout in seconds
            if(res):
                for voice, entry in res.items():
                    if(entry):
                        comb_key = f"{key}_{voice}"
                        results[comb_key] = entry
            else:
                failed+=1


    # Filter out empty results

    print(f"Onset extraction failed for {failed} out of {len(results)} files")
    return results


if __name__=="__main__":
    with open(ONSET_FILE_NAME,"rb") as pfile:
        onset_data = pickle.load(pfile)
    onsets_labels = process_abc_dict(onset_data)
    print(onsets_labels)
    df = pd.DataFrame.from_dict(onsets_labels, orient='index')
    df.to_csv(OUTPUT_FILE_NAME)
    
        

