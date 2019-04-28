# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:10:14 2019

@author: em370_000
"""

my_numbers = []
my_words = []


with open('eval.txt','r') as f:
    for line in f:
        words = line.split()
        if len(words)>0:
            my_numbers.append(words[0])
            my_words.append(words[1])
            
            
for number,word in zip(my_numbers,my_words):
    print("{} {} \n".format(number,word))