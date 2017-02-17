#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import os
import glob


list_dirs = {
    'train': 'images/train/*',
    'test': 'images/test/*'
}

for file_name, file_dir in list_dirs.items():
    text_file = open('{}.txt'.format(file_name), 'w')
    answer_dirs = glob.glob(file_dir)
    for answer_dir in answer_dirs:
        answer = os.path.basename(answer_dir)
        image_file_paths = glob.glob('{}/*'.format(answer_dir))
        for image_file_path in image_file_paths:
            line = '{}\n'.format(' '.join([image_file_path, answer]))
            text_file.write(line)
    text_file.close()
