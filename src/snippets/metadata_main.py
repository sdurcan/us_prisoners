# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:53:57 2022

@author: siobh
"""
#from src import snippets

import generate_metadata as gm

metadata=gm.generate_metadata()

metadata.load_metadata()
#metadata.setup_dict()
#metadata.calculate_var_features()