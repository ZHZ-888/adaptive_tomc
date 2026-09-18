#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:21:42 2024

@author: zzha
"""
'''
print_control.py - define a global print switch
In Python, it is customary to use all uppercase letters to name global \
    constants.
'''
PRINT_ENABLED = True

def print_message(*args):
    if PRINT_ENABLED:
        print(*args)
