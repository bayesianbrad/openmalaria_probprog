#!/usr/bin/env bash
docker build -t bradleygh/openmalariapprun .
if [ $? -eq 0 ]; then
  docker push bradleygh/openmalariapprun
fi
