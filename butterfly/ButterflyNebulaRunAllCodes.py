#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:01:01 2024

@author: nclark
"""

# this python file will run all of my analysis codes and then plotting codes sequentially

import subprocess

subprocess.run("ButterflyNebulaAnalysis_prep.py", shell=True)
subprocess.run("ButterflyNebulaAnalysis.py", shell=True)
subprocess.run("ButterflyNebulaFigs.py", shell=True)