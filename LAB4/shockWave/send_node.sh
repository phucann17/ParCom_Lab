#!/bin/bash

# send_node.sh - chỉ copy đến MPI-node9 dùng scp


NODE="MPI-node9"

SRC_DIR="/root/LAB4/ParCom_Lab/LAB4"

DEST_DIR="/root/LAB4/ParCom_Lab/LAB4"


echo "Copying to $NODE..."


# Tạo thư mục đích nếu chưa tồn tại

ssh root@"$NODE" "mkdir -p ${DEST_DIR}"


# Copy tất cả file và thư mục, bỏ *.csv và .git

cd "$SRC_DIR" || exit


# Copy các thư mục (trừ .git)

for dir in */ ; do

if [[ "$dir" != ".git/" ]]; then

scp -r "$dir" root@"$NODE":"$DEST_DIR/"

fi

done

# Copy các file (trừ *.csv và file git)

for file in * ; do

if [[ -f "$file" && "$file" != *.csv && "$file" != ".gitignore" && "$file" != ".gitattributes" ]]; then

scp "$file" root@"$NODE":"$DEST_DIR/"

fi

done


echo "Done copying to $NODE."