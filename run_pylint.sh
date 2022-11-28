#!/bin/bash

# remove trailing whitespaces
find cmethods -type f -name '*.py' -exec sed -i '' 's/[[:space:]]*$//' {} \+

# run pylint
pylint --rcfile=.pylintrc cmethods 