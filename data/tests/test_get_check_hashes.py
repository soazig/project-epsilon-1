""" Tests for checking the hashes for downloaded data
Run with:
    nosetests test_get_check_hashes.py
"""
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
import sys, os, pdb
import tempfile
import json
#Specicy the path for functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from data_hashes import *

import urllib.request

def test_check_hashes():
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(b'Some data')
        temp.flush()
        fname = temp.name
        d = {fname: "5b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert check_hashes(d)
        d = {fname: "4b82f8bf4df2bfb0e66ccaa7306fd024"}
        assert not check_hashes(d)


def test_get_hashes():
    #Download json file senators-list.json and store it in the current directory
    url = 'http://jarrodmillman.com/rcsds/data/senators-list.json'
    with urllib.request.urlopen(url) as in_file,\
        open('senators-list.json', 'wb') as out_file:
        data = in_file.read() # a `bytes` object
        out_file.write(data)
    dir_hashes = generate_dir_md5('.')
    #Verify the hash for senators-list.json and remove the file
    assert not dir_hashes['./senators-list.json'] == \
              'f0c1a76b571ab86968329a7b202f2edf'
    assert dir_hashes['./senators-list.json'] == \
              'd96aaf10750b3f44303dd055d7868b2d'
    os.remove('senators-list.json')
