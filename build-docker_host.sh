#!/usr/bin/env bash
docker build OMhost.Dockerfile -t bradleygh/openmalariapp.
if [ $? -eq 0 ]; then
  docker push bradleygh/openmalariapp
fi
