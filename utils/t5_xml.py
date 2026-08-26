""" 
    Utility script to convert MelodyT5 dataset from ABC format to MusicXML format.
"""
from .convabc2xml import convert_abc2xml
import argparse

# convert to musicxml
def main(t5_abc: str, t5_xml: str):
    convert_abc2xml(t5_abc, t5_xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_abc', default="/home/nkcemeka/Documents/Datasets/MelodyT5/t5_abc")
    parser.add_argument('--t5_xml', default="/home/nkcemeka/Documents/Datasets/MelodyT5/t5_xml")
    args = parser.parse_args()
    main(t5_abc=args.t5_abc, t5_xml=args.t5_xml)
