#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:55:57 2020

@author: vws
"""

class Alphabet(object):
    
    def __init__(self, alphabet_file: str):
        
        with open(alphabet_file, encoding="utf-8") as f:
            
            # Encoding dictionary
            self.character_map = {k: int(v) for line in f for (k, v) in (line.strip().split(None, 1),)}
            
            # Decoding dictionary
            self.label_map     = {v: k for k, v in self.character_map.items()}
        
        # Hard-coded, no solution yet
        self.label_map[1] = ' '
        
    def encode(self, text: str):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.character_map['<SPACE>']
            else:
                ch = self.character_map[c]
            int_sequence.append(ch)
        return int_sequence
        
    def decode(self, labels: int):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.label_map[i])
        return ''.join(string).replace('', ' ')