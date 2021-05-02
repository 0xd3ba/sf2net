import os
import json
import pathlib


def read_json(file):
    """ Reads the json file as a dictionary and returns it """
    fpath = pathlib.Path(file)
    with fpath.open('r') as file:
        json_dict = json.load(file)

    return json_dict
