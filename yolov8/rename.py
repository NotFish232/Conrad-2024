#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# rename_events.py

import sys
from pathlib import Path
import os
# Use this if you want to avoid using the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event

def rename_events(input_path, output_path, old_tags, new_tag):
    # Make a record writer
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([str(input_path)]):
            # Read event
            ev = Event()
            ev.MergeFromString(rec.numpy())
            # Check if it is a summary
            if ev.summary:
                # Iterate summary values
                for v in ev.summary.value:
                    # Check if the tag should be renamed
                    if v.tag in old_tags:
                        # Rename with new tag name
                        v.tag = new_tag
            writer.write(ev.SerializeToString())

def rename_events_dir(input_dir, output_dir, old_tags, new_tag):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Make output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # Iterate event files
    for ev_file in input_dir.glob('**/*.tfevents*'):
        # Make directory for output event file
        out_file = Path(output_dir, ev_file.relative_to(input_dir))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Write renamed events
        rename_events(ev_file, out_file, old_tags, new_tag)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f'{sys.argv[0]} <input dir> <output dir> <old tags> <new tag>',
              file=sys.stderr)
        sys.exit(1)
    input_dir, output_dir, old_tags, new_tag = sys.argv[1:]
    old_tags = old_tags.split(';')
    rename_events_dir(input_dir, output_dir, old_tags, new_tag)
    print('Done')