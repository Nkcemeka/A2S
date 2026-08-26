import os
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

def convert_file_abc2xml(abc_file: str, out_file: str):
    """
        Converts xml file to abc.

        Args:
            abc_file (str): Name of abc file.
            out_file (str): Name of musicxml file.
    """
    print(f"Converting abc file to MusicXML!")
    base_path = Path(__file__).parent
    cmd = f'python {base_path}/abc2xml.py '
    file = abc_file

    try:
        p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
        result = p.communicate()
        output = result[0].decode('utf-8')

        if output == '':
            return
        else:
            store_path = out_file
            with open(store_path, 'w', encoding='utf-8') as f:
                f.write(output)
    except Exception as e:
        return


def convert_abc2xml(BASE_FOLDER: str, XML_FOLDER: str):
    print(f"Converting abc to MusicXML!")
    base_path = Path(__file__).parent
    cmd = f'python {base_path}/abc2xml.py '
    file_dir = Path(BASE_FOLDER)
    file_list = sorted(file_dir.rglob("*.abc"))

    for file in tqdm(file_list):
        file = str(file)
        filename = file.split('/')[-1]  # Extract file name
        os.makedirs(XML_FOLDER, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if output == '':
                continue
            else:
                output_path = f"{XML_FOLDER}/" + ".".join(filename.split(".")[:-1]) + ".musicxml"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            continue
