from pathlib import Path
import os
import random
import subprocess
from tqdm import tqdm
import re
from abctoolkit.utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from multiprocessing import Pool

def convert_file_xml2abc(xml_file: str, out_file: str):
    """
        Converts xml file to abc.

        Args:
            xml_file (str): Name of xml file.
            out_file (str): Name of abc file.
    """
    print(f"Converting MusicXML file to abc!")
    base_path = Path(__file__).parent
    cmd = f'python {base_path}/xml2abc.py -d 8 -c 6 -x '
    file = xml_file
    filename = Path(file).stem

    try:
        p = subprocess.Popen(cmd + '"' + str(file) + '"', stdout=subprocess.PIPE, shell=True)
        result = p.communicate()
        output = result[0].decode('utf-8')

        if output == '':
            return
        else:
            store_path = out_file
            with open(store_path, "w", encoding="utf-8") as f:
                f.write(output)
    except Exception as e:
        return

def convert_xml2abc(BASE_FOLDER: str, ABC_FOLDER: str):
    """
        Converts xml to abc.

        Args:
            BASE_FOLDER (str): Folder containing musicxml files.
            ABC_FOLDER (str): Folder to save abc file.
    """
    print(f"Converting MusicXML to abc!")
    Path(ABC_FOLDER).mkdir(exist_ok=True, parents=True)
    base_path = Path(__file__).parent
    cmd = f'python {base_path}/xml2abc.py -d 8 -c 6 -x '
    file_dir = Path(BASE_FOLDER)
    file_list = sorted(file_dir.rglob("*.musicxml"))
    for file in tqdm(file_list):
        filename = file.stem
        try:
            p = subprocess.Popen(cmd + '"' + str(file) + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if output == '':
                continue
            else:
                store_path = Path(ABC_FOLDER) / f"{filename}.abc"
                with open(store_path, "w", encoding="utf-8") as f:
                    f.write(output)
        except Exception as e:
            continue

def abc_preprocess_pipeline(abc_path, INTERLEAVED_FOLDER, suffix: str|None=None):

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # delete blank lines
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode
    abc_lines = unidecode_abc_lines(abc_lines)

    # clean information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # delete bar number annotations
    abc_lines = remove_bar_no_annotations(abc_lines)

    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            print(abc_path, 'Unequal bar number')
            raise Exception
    except:
        raise Exception

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]

    # transpose
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    if global_metadata_dict['K'][0] == 'none':
        global_metadata_dict['K'][0] = 'C'

    interleaved_abc = rotate_abc(abc_lines)

    if suffix is None:
        interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + '.abc')
    else:
        interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + f'_{suffix}.abc')
    with open(interleaved_path, 'w') as w:
        w.writelines(interleaved_abc)

def interleaved_process(BASE_FOLDER: str, OUTPUT_DIR: str):
    INTERLEAVED_FOLDER = str(Path(OUTPUT_DIR))
    file_dir = Path(BASE_FOLDER)
    file_list = sorted(file_dir.rglob("*.abc"))
    print(f"Converting Standard ABC to Interleaved ABC!")
    for f in tqdm(file_list):
        abc_preprocess_pipeline(str(f), INTERLEAVED_FOLDER)

def interleaved_process_file(abc_file: str, INTERLEAVED_FOLDER, suffix: str|None=None):
    print(f"Converting Standard ABC to Interleaved ABC!")
    abc_preprocess_pipeline(abc_file, INTERLEAVED_FOLDER, suffix=suffix)
