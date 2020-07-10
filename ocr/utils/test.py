
import os
import tarfile
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import logging

from mxnet.gluon.data import dataset
from mxnet import nd

# def __init__():
#     print("hello")

# __init__()

# from .expand_bounding_box import expand_bounding_box
def __init__(self, credentials=None,
                 root=os.path.join(os.path.dirname(__file__), '..', '..','dataset', 'iamdataset'), 
                 train=True, output_data="text",
                 output_parse_method=None,
                 output_form_text_as_array=False):
    print("hello")

#         _parse_methods = ["form", "form_original", "form_bb", "line", "word"]
#         error_message = "{} is not a possible parsing method: {}".format(
#             parse_method, _parse_methods)
#         assert parse_method in _parse_methods, error_message
#         self._parse_method = parse_method
#         url_partial = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/{data_type}/{filename}.tgz"
#         if self._parse_method == "form":
#             self._data_urls = [url_partial.format(data_type="forms", filename="forms" + a) for a in ["A-D", "E-H", "I-Z"]]
#         elif self._parse_method == "form_bb":
#             self._data_urls = [url_partial.format(data_type="forms", filename="forms" + a) for a in ["A-D", "E-H", "I-Z"]]
#         elif self._parse_method == "form_original":
#             self._data_urls = [url_partial.format(data_type="forms", filename="forms" + a) for a in ["A-D", "E-H", "I-Z"]]
#         elif self._parse_method == "line":
#             self._data_urls = [url_partial.format(data_type="lines", filename="lines")]
#         elif self._parse_method == "word":
#             self._data_urls = [url_partial.format(data_type="words", filename="words")]
#         self._xml_url = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz"

#         if credentials == None:
#             if os.path.isfile(os.path.join(os.path.dirname(__file__), '.', 'credentials.json')):
#                 with open('credentials.json') as f:
#                     credentials = json.load(f)
#                 self._credentials = (credentials["username"], credentials["password"])
#                 print(credentials)
#             else:
#                 # assert False, "Hello"
#                 print("Hello")
#                 # print(credentials["username"])
#         else:
#             self._credentials = credentials
        
#         self._train = train

#         _output_data_types = ["text", "bb"]
#         error_message = "{} is not a possible output data: {}".format(
#             output_data, _output_data_types)
#         assert output_data in _output_data_types, error_message
#         self._output_data = output_data

#         if self._output_data == "bb":
#             assert self._parse_method in ["form", "form_bb"], "Bounding box only works with form."
#             _parse_methods = ["form", "line", "word"]
#             error_message = "{} is not a possible output parsing method: {}".format(
#                 output_parse_method, _parse_methods)
#             assert output_parse_method in _parse_methods, error_message
#             self._output_parse_method = output_parse_method

#             self.image_data_file_name = os.path.join(root, "image_data-{}-{}-{}*.plk".format(
#                 self._parse_method, self._output_data, self._output_parse_method))
#         else:
#             self.image_data_file_name = os.path.join(root, "image_data-{}-{}*.plk".format(self._parse_method, self._output_data))

#         self._root = root
#         if not os.path.isdir(root):
#             os.makedirs(root)
#         self._output_form_text_as_array = output_form_text_as_array
        
#         data = self._get_data()
#         # super(IAMDataset, self).__init__(data)




train_ds = __init__("form", output_data="bb", output_parse_method="form", train=True)
# print("Number of training samples: {}".format(len(train_ds)))