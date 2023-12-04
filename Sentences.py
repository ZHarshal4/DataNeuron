# -*- coding: utf-8 -*-
"""
@author: HP
"""
from pydantic import BaseModel
# Class for 2 Sentences
class sentences(BaseModel):
    text1: str 
    text2: str