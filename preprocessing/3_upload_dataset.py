from datasets import Dataset
import pandas as pd
import ast
dataset_path = "/home/efraim/Desktop/RagtimeExp/labeled_onsets_tiny.csv"

df = pd.read_csv(dataset_path)

df = df.set_index("Unnamed: 0")
df.index.name = "index"
GRID_SIZE = 48


hf_dataset = Dataset.from_pandas(df)


# Function to parse both columns
def parse_both_columns(example):

    def eval_lis(s):
        l=ast.literal_eval(s)
        return l

    def parse_and_sort(s):
        d = ast.literal_eval(s)
        bars = [[0]*GRID_SIZE]*len(d)
        for i in range(0,len(d)):
            bar_eval = d[i]
            if(len(bar_eval)<GRID_SIZE):
                bar_eval+=[0]*(GRID_SIZE-len(bar_eval))
            elif(len(bar_eval)>GRID_SIZE):
                bar_eval=bar_eval[:GRID_SIZE]
            bars[i]=bar_eval
        return [sorted(list(d[k])) for k in sorted(d)]


    return {
        "metric_vector": parse_and_sort(example["metric_arranged"]),
        "spectral_vector": parse_and_sort(example["spectral_arranged"]),
        "syncopation": eval_lis(example["syncopation"]),
        "distances": eval_lis(example["distances"])
    }

# Apply function
hf_dataset = hf_dataset.map(parse_both_columns)

# Optional: remove original string columns
hf_dataset = hf_dataset.remove_columns(["metric_arranged", "spectral_arranged"])

# Optionally remove the automatically added index column
hf_dataset.push_to_hub("efraimdahl/syncopation-dataset")