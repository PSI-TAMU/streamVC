import glob
from pathlib import Path

if __name__ == "__main__":
    print("Processing..")
    data_files = glob.glob("/mnt/nvme-data1/waris/streamVC_data/data/**/wav/*.wav")
    data_tsv_fpath = Path("/mnt/nvme-data1/waris/streamVC_data/metadata")
    
    data_files = [ f'{data_file}\n' for data_file in data_files ]
    open((data_tsv_fpath / 'data.tsv'), 'w').writelines(data_files)

    print("Done!")
