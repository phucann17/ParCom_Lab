#!/bin/bash

SRC_DIR="$HOME/ParCom/ParCom_Lab"
DEST_USER="admin1"
DEST_HOST="192.168.1.15"
DEST_DIR="/home/admin1/ParCom/ParCom_Lab"

echo "Copying $SRC_DIR → $DEST_USER@$DEST_HOST:$DEST_DIR"
echo "Excluding: *.csv + all .git files"
echo

ssh ${DEST_USER}@${DEST_HOST} "mkdir -p ${DEST_DIR}"

rsync -av \
    --exclude="*.csv" \
    --exclude=".git/" \
    --exclude=".git/**" \
    --exclude=".gitignore" \
    --exclude=".gitattributes" \
    "${SRC_DIR}/" \
    "${DEST_USER}@${DEST_HOST}:${DEST_DIR}/"

echo
echo "DONE!"
