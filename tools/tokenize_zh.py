#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import jieba

if __name__ == "__main__":
    text_path = sys.argv[1]
    save_path = sys.argv[2]
    assert os.path.isfile(text_path)
    obj = jieba.Tokenizer()
    obj.tmp_dir = "jieba_cache"
    with open(text_path,'r',encoding='utf-8') as f_in,open(save_path,'w',encoding='utf-8') as f_out:
        while True:
            line = f_in.readline()
            if not line:
                break
            f_out.write(' '.join(obj.cut(line.lstrip().rstrip())) + "\n" )
