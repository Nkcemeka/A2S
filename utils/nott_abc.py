""" 
    Utility script to convert Nottingham dataset from XML format to ABC format.
"""
import argparse
from .convxml2abc import convert_xml2abc

def main(xml_dir: str, abc_dir: str):
    convert_xml2abc(xml_dir, abc_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', \
        default="/home/nkcemeka/Documents/Datasets/Nottingham/nott_xml")
    parser.add_argument('--abc_dir', \
        default="/home/nkcemeka/Documents/Datasets/Nottingham/nott_abc")

    args = parser.parse_args()
    main(xml_dir=args.xml_dir, abc_dir=args.abc_dir)

