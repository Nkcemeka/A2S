import pretty_midi
import mido
import subprocess
import tempfile
import numpy as np
import shutil
from pathlib import Path
import os
import music21 as m21
from collections import defaultdict
import random
from copy import deepcopy
from .convxml2abc import interleaved_process_file, convert_file_xml2abc
import soundfile as sf
import re
from abctoolkit.utils import Barline_regexPattern
from abctoolkit.duration import calculate_bartext_duration

def rest_unreduce(abc_lines):
    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}  
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass    # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


# Linked List process for dealing with ties...
class LinkedList:
    def __init__(self, head):
        self.head = head
        self.last = head
    
    def append(self, node):
        self.last.next = node
        self.last = node

class Node:
    def __init__(self, el, next=None):
        self.el = el
        self.next = next
    
    def __str__(self):
        return f"{self.el.nameWithOctave}"

    def __repr__(self):
        return self.__str__()

def remove_broken_ties(s: m21.stream.Stream):
    broken_ties = []
    tie_dict = defaultdict(LinkedList)
    for el in list(s.recurse()):
        if isinstance(el, m21.note.Note) or isinstance(el, m21.chord.Chord):
            notes = []
            if isinstance(el, m21.chord.Chord):
                for n in el.notes:
                    notes.append(n)
            else:
                notes.append(el)
            
            for n in notes:
                if n.tie is None:
                    continue
                tie_type = n.tie.type

                if tie_type == "start":
                    tie_dict[n.nameWithOctave] = LinkedList(Node(n))
                
                if tie_type == "continue":
                    ll = tie_dict.get(n.nameWithOctave, None)
                    if ll is None:
                        # This is a broken tie
                        broken_ties.append(n)
                    else:
                        ll.append(Node(n))
                
                if tie_type == "stop":
                    ll = tie_dict.get(n.nameWithOctave, None)
                    if ll is None:
                        # This is a broken tie
                        broken_ties.append(n)
                    else:
                        ll.append(Node(n))

    # Now we have to pass the dict to add elements that have a start tie but no end tie
    for key in tie_dict.keys():
        ll = tie_dict[key]
        last_node = ll.last
        curr_node = ll.head
        if last_node.el.tie is None:
            continue
        else:
            tie_type = last_node.el.tie.type
            if tie_type != "stop":
                # Add all of the notes from the head node
                # to broken ties list
                while curr_node:
                    broken_ties.append(curr_node.el)
                    curr_node = curr_node.next
                    if curr_node is None:
                        break

    for n in broken_ties:
        n.tie = None
    return s

def midi2audio(midi_obj, sf2_path: str, fs=16000, prog: str="piano") -> np.ndarray:
    """  
        Convert MIDI to audio

        Args:
            midi_obj: prettyMIDI | Mido object 
            fs: sampling frequency/rate
            sf2_path: Path to soundfont file

        Return:
            audio_data (np.ndarray): rendered audio data
    """
    # Convert MIDI to audio
    if isinstance(midi_obj, mido.MidiFile):
        # can only do this in > version 0.2.11 of pretty_midi
        midi_obj = pretty_midi.PrettyMIDI(mido_object=midi_obj)

    programs = {"piano": "Acoustic Grand Piano", 
                "alto_sax": "Alto Sax", "tenor_sax": "Tenor Sax", 
                "sax": random.choice(["Alto Sax", "Tenor Sax"]), 
                "violin": "Violin"}

    for instrument in midi_obj.instruments:
        instrument.program = pretty_midi.instrument_name_to_program(programs[prog])

    audio_data = midi_obj.fluidsynth(fs=fs, \
                sf2_path=sf2_path)
    return audio_data

def convert_musescore(input_path:str, output_path: str, MUSESCORE_PATH: str, LD_PATH: str=""):
    """ 
        Performs conversions from one file to another
        using the Musescore application.

        Args:
            input_path (str): Path to input file to convert
            output_path (str): Output path to convert to
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = str(Path(input_path).suffix)[1:] # we start from 1 to remove the dot
        shutil.copy(input_path, f"{tmpdir}/temp.{suffix}")

        # Update Environment Variables
        evs = os.environ.copy()
        evs["DISPLAY"] = ":0"
        evs["QT_QPA_PLATFORM"] = "offscreen"
        evs["XDG_RUNTIME_DIR"] = tmpdir
        evs["LD_LIBRARY_PATH"] = LD_PATH + evs.get("LD_LIBRARY_PATH", "")

        # perform conversion in a new environment
        subprocess.run(
            [MUSESCORE_PATH, "-o",  output_path, f"{tmpdir}/temp.{suffix}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=evs,
            check=True
        )

def strip_repeats(score: m21.stream.Stream) -> m21.stream.Stream:
    for b in score.recurse().getElementsByClass(m21.bar.Repeat):
        # Get the measure containing the repeat barline
        measure = b.activeSite
        if b.direction == 'end':
            measure.rightBarline = m21.bar.Barline('regular')
        elif b.direction == 'start':
            measure.leftBarline = m21.bar.Barline('regular')

    # Remove text-based repeat expressions (e.g., D.C. al Fine)
    for expr in score.recurse().getElementsByClass(m21.repeat.RepeatExpression):
        expr.activeSite.remove(expr)

    return score

def windowed_segments(score_file: str, window_size: int, hop_size: int, out_dir_xml: str, \
        out_dir_midi: str|None=None, out_dir_audio:str|None=None, \
        out_dir_abc:str|None=None, out_dir_abc_il:str|None=None,\
        flag_midi: bool=False, flag_abc: bool=False, flag_abc_il: bool=False, \
        flag_audio: bool=False, sample_rate: float=24000, MUSESCORE_PATH:str|None=None, \
        prog: str="piano", sf2_path:str|None = None, \
        strip_flag: bool=True, part_idx: int|None=None):
    """
        Created windowed score segments.
    """
    # Create out_dir_score, just in case
    Path(out_dir_xml).mkdir(exist_ok=True, parents=True)

    # Load the score and expand the repeats (if necessary)
    suffix = Path(score_file).suffix
    if suffix == ".krn":
        with open(str(score_file), "r", encoding="utf-8") as f:
            humdrum_data = f.read()
            score = m21.converter.parseData(humdrum_data, \
                format="humdrum")
    else:
        score = m21.converter.parse(score_file)

    if strip_flag:
        score = strip_repeats(score)
    else:
        try:
            score = score.expandRepeats()
        except:
            print(f"Skippping bad score...")
            return

    # Iterate over each part independently
    parts = score.parts
    if part_idx is not None:
        assert part_idx < len(parts), \
            f"Requested part {part_idx}, but score has {len(parts)} parts."
        parts = [parts[part_idx]]

    # Number of measures (assumes all parts have same measure count)
    # Extract measures for each selected part
    measures = [
        list(part.getElementsByClass(m21.stream.Measure))
        for part in parts
    ]

    # All selected parts must have the same number of measures
    total_measures = len(measures[0])

    for part_measures in measures[1:]:
        assert len(part_measures) == total_measures, \
            "Number of measures for selected parts should be equal."
    
    total_duration = 0

    for start in range(0, total_measures - window_size + 1, hop_size):
        print(f"{total_duration/3600} hours processed (xml and all)...")
        
        excerpt = m21.stream.Score()
        excerpt.metadata = deepcopy(score.metadata)

        for part in parts:
            measures = list(part.getElementsByClass(m21.stream.Measure))
            window = measures[start:start + window_size]

            if not window:
                continue

            new_part = m21.stream.Part()

            # copy the measures first
            for i, measure in enumerate(window, start=1):
                m = deepcopy(measure)
                m.number = i
                new_part.append(m)

            # Determine the context that is active at the beginning
            # of the original first measure in the window.
            original_first = window[0]
            new_first = new_part.measure(1)

            for cls in (
                m21.clef.Clef,
                m21.key.KeySignature,
                m21.meter.TimeSignature,
                m21.tempo.MetronomeMark,
            ):
                obj = original_first.getContextByClass(cls)
                if obj is not None:
                    new_first.insert(0, deepcopy(obj))

            excerpt.insert(0, new_part)

        # We probably do not need to worry about broken ties here or do we??
        excerpt = remove_broken_ties(excerpt)
        out_file = str(Path(out_dir_xml) / (Path(score_file).stem +f"_{start}.musicxml"))
        try:
            assert excerpt.isWellFormedNotation(), "Notation not well formed!"
            excerpt_dur = excerpt.secondsMap[-1]["endTimeSeconds"]
            excerpt.write("musicxml", str(out_file))

            if flag_midi:
                Path(out_dir_midi).mkdir(parents=True, exist_ok=True)
                out_file_midi = str(Path(out_dir_midi) / (Path(score_file).stem + f"_{start}.mid"))
                if MUSESCORE_PATH is None:
                    raise ValueError("Musescore path should not be none.")
                convert_musescore(out_file, out_file_midi, MUSESCORE_PATH=MUSESCORE_PATH)

                if flag_audio and Path(out_file_midi).exists():
                    if sf2_path is None:
                        raise ValueError("Please supply the soundfont path.")

                    Path(out_dir_audio).mkdir(parents=True, exist_ok=True)
                    out_file_audio = str(Path(out_dir_audio) / (Path(score_file).stem + f"_{start}.wav"))
                    midi_obj = pretty_midi.PrettyMIDI(out_file_midi)
                    audio = midi2audio(midi_obj, prog=prog, \
                        sf2_path=sf2_path, fs=sample_rate)
                    sf.write(out_file_audio, audio, samplerate=sample_rate)

            if flag_abc:
                Path(out_dir_abc).mkdir(parents=True, exist_ok=True)
                out_file_abc = str(Path(out_dir_abc) / (Path(score_file).stem + f"_{start}.abc"))
                convert_file_xml2abc(out_file, out_file_abc)

                # load the abc file and clean it of its tempo markings
                abc_text = Path(f"{out_file_abc}").read_text(encoding="utf-8")
                abc_text = postprocess_abc(abc_text)
                with open(str(out_file_abc), "w") as f:
                    f.write(abc_text)

                if flag_abc_il:
                    Path(out_dir_abc_il).mkdir(parents=True, exist_ok=True)
                    interleaved_process_file(out_file_abc, out_dir_abc_il)
                
            total_duration += excerpt_dur
        except Exception as e:
            print(f"{e}")
            continue

def get_measure_duration_seconds(measure, default_bpm=120):
    tempo = measure.getContextByClass(m21.tempo.MetronomeMark)

    if tempo is None:
        bpm = default_bpm
    else:
        bpm = tempo.number

    return measure.duration.quarterLength * 60.0 / bpm

def windowed_segments_cumulative(score_file: str, out_dir_xml: str, \
        out_dir_midi: str|None=None, out_dir_audio:str|None=None, \
        out_dir_abc:str|None=None, out_dir_abc_il:str|None=None,\
        flag_midi: bool=False, flag_abc: bool=False, flag_abc_il: bool=False, \
        flag_audio: bool=False, sample_rate: float=24000, MUSESCORE_PATH:str|None=None, \
        prog: str="piano", sf2_path:str|None = None, \
        strip_flag: bool=True, part_idx: int|None=None, audio_length: float=20):
    """
        Created windowed score segments.
    """
    # Create out_dir_score, just in case
    Path(out_dir_xml).mkdir(exist_ok=True, parents=True)

    # Load the score and expand the repeats (if necessary)
    suffix = Path(score_file).suffix
    if suffix == ".krn":
        with open(str(score_file), "r", encoding="utf-8") as f:
            humdrum_data = f.read()
            score = m21.converter.parseData(humdrum_data, \
                format="humdrum")
    else:
        score = m21.converter.parse(score_file)

    if strip_flag:
        score = strip_repeats(score)
    else:
        try:
            score = score.expandRepeats()
        except:
            print(f"Skippping bad score...")
            return

    # Iterate over each part independently
    parts = score.parts
    if part_idx is not None:
        assert part_idx < len(parts), \
            f"Requested part {part_idx}, but score has {len(parts)} parts."
        parts = [parts[part_idx]]

    # Number of measures (assumes all parts have same measure count)
    # Extract measures for each selected part
    measures = [
        list(part.getElementsByClass(m21.stream.Measure))
        for part in parts
    ]

    # All selected parts must have the same number of measures
    total_measures = len(measures[0])

    for part_measures in measures[1:]:
        assert len(part_measures) == total_measures, \
            "Number of measures for selected parts should be equal."
    
    total_duration = 0
    start = 0

    while start < total_measures:
        print(f"{total_duration/3600} hours processed (xml and all)...")
        
        excerpt = m21.stream.Score()
        excerpt.metadata = deepcopy(score.metadata)

        for idx, part in enumerate(parts):
            measures_part = measures[idx][start:]
            new_part = m21.stream.Part()

            # copy the measures that fit the given number of seconds
            total_secs = 0
            for i, window in enumerate(measures_part):
                window_secs = get_measure_duration_seconds(window)
                exp_dur = window_secs + total_secs # expected duration of excerpt
                if (exp_dur <= audio_length):
                    m = deepcopy(window)
                    m.number = i # or i + 1??? Does it really matter?
                    new_part.append(m)
                    total_secs += window_secs

            if total_secs < 5:
                # This is not a valid musical coherent excerpt
                # so we discard it...
                break

            # Determine the context that is active at the beginning
            # of the original first measure in the window.
            original_first = measures_part[0]
            new_first = new_part.measure(1)

            for cls in (
                m21.clef.Clef,
                m21.key.KeySignature,
                m21.meter.TimeSignature,
                m21.tempo.MetronomeMark,
            ):
                obj = original_first.getContextByClass(cls)
                if obj and new_first:
                    new_first.insert(0, deepcopy(obj))

            excerpt.insert(0, new_part)

        excerpt_measures = len(excerpt.parts[0].getElementsByClass("Measure"))
        if excerpt_measures == 0:
            # we added nothing
            start += 1
            continue
        else:
            start += excerpt_measures

        # We probably do not need to worry about broken ties here or do we??
        excerpt = remove_broken_ties(excerpt)
        out_file = str(Path(out_dir_xml) / (Path(score_file).stem +f"_{start}.musicxml"))
        try:
            assert excerpt.isWellFormedNotation(), "Notation not well formed!"
            excerpt_dur = excerpt.secondsMap[-1]["endTimeSeconds"]
            excerpt.write("musicxml", str(out_file))

            if flag_midi:
                Path(out_dir_midi).mkdir(parents=True, exist_ok=True)
                out_file_midi = str(Path(out_dir_midi) / (Path(score_file).stem + f"_{start}.mid"))
                if MUSESCORE_PATH is None:
                    raise ValueError("Musescore path should not be none.")
                convert_musescore(out_file, out_file_midi, MUSESCORE_PATH=MUSESCORE_PATH)

                if flag_audio and Path(out_file_midi).exists():
                    if sf2_path is None:
                        raise ValueError("Please supply the soundfont path.")

                    Path(out_dir_audio).mkdir(parents=True, exist_ok=True)
                    out_file_audio = str(Path(out_dir_audio) / (Path(score_file).stem + f"_{start}.wav"))
                    midi_obj = pretty_midi.PrettyMIDI(out_file_midi)
                    audio = midi2audio(midi_obj, prog=prog, \
                        sf2_path=sf2_path, fs=sample_rate)

                    audio_length_samples = audio_length * sample_rate
                    if len(audio) < audio_length_samples:
                        padding = audio_length_samples - len(audio)
                        audio = np.pad(audio, (0, padding), 'constant')
                    sf.write(out_file_audio, audio, samplerate=sample_rate)

            if flag_abc:
                Path(out_dir_abc).mkdir(parents=True, exist_ok=True)
                out_file_abc = str(Path(out_dir_abc) / (Path(score_file).stem + f"_{start}.abc"))
                convert_file_xml2abc(out_file, out_file_abc)

                # load the abc file and clean it of its tempo markings
                abc_text = Path(f"{out_file_abc}").read_text(encoding="utf-8")
                abc_text = postprocess_abc(abc_text)
                with open(str(out_file_abc), "w") as f:
                    f.write(abc_text)

                if flag_abc_il:
                    Path(out_dir_abc_il).mkdir(parents=True, exist_ok=True)
                    interleaved_process_file(out_file_abc, out_dir_abc_il)
                
            total_duration += excerpt_dur
        except Exception as e:
            print(f"{e}")
            continue
        
def postprocess_abc(abc: str):
    lines = abc.splitlines()
    clean_lines = []

    for line in lines:
        if line.startswith("Q:"):
            continue

        if line.startswith("M:") and "none" in line:
            line = "M:4/4"

        if line.startswith("K:") and "none" in line:
            line = "K:C"

        if line.startswith("V:"):
            line = re.sub(r'\s+nm="[^"]*"', '', line)
            line = re.sub(r'\s+snm="[^"]*"', '', line)

        # Remove chord annotations such as "G"
        line = re.sub(r'"[^"]*"', '', line)

        clean_lines.append(line)

    return '\n'.join(clean_lines)
