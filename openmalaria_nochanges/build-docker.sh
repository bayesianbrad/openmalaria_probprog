#!/usr/bin/env bash
docker build -t bradleygh/openmalaria .
if [ $? -eq 0 ]; then
  docker push bradleygh/openmalaria
fi
