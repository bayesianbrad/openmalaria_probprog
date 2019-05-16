#!/usr/bin/env bash
docker build -t bradleygh/openmalariapp .
if [ $? -eq 0 ]; then
  docker push bradleygh/openmalariapp
fi
