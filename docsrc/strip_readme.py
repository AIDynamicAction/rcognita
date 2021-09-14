#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os


target = os.path.abspath(__file__ + "/../../README.rst")
destination = os.path.abspath(__file__ + "/../README.rst")

with open(target, "r") as f:
    s = f.read()
    

logo_top = ".. image:: https://raw.githubusercontent.com/AIDynamicAction/rcognita/d0c7d1173b51e0ed5df044cf1fb1c92eca53d819/gfx/logo/rcognita-logo.png\n"
m = re.search(f"({logo_top}.*?\n)[a-zA-Z0-9\s]*?\n=", s, re.S)
logo = m.group(1)

m = re.search("(Example run with a mobile robot simulation\n.*?\n)[a-zA-Z0-9\s]*?\n=", 
              s, flags=re.S)
example = m.group(1)


m = re.search("(Table of content\n.*?\n)[a-zA-Z0-9\s]*?\n=",
              s, flags=re.S)
table_of_content = m.group(1)


links_to_table = '`To table of content <#Table-of-content>`__'


link_to_docs = 'A detailed documentation is available `here <https://aidynamicaction.github.io/rcognita/>`__.'


for fragment in [logo,
                 example,
                 table_of_content,
                 links_to_table,
                 link_to_docs]:
    s = s.replace(fragment, "")


with open(destination, "w") as f:
    f.write(s)
