#!/bin/bash

OUTPUT_TAR="submission.tar.gz"
FILES_TO_INCLUDE="main.py utils"
tar -czf $OUTPUT_TAR $FILES_TO_INCLUDE
echo "Packaged files into $OUTPUT_TAR"