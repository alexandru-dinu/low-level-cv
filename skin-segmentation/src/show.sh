#!/bin/sh

case "$1" in
    "img_1")
        python main.py --img_path ../data/img_1.jpg --low_thr 5 30 150 --high_thr 20 150 255 --blob_thr 5000
        ;;
    "img_2")
        python main.py --img_path ../data/img_2.jpg --low_thr 5 50 150 --high_thr 20 110 255 --blob_thr 5000
        ;;
    "img_3")
        python main.py --img_path ../data/img_3.jpg --low_thr 5 50 150 --high_thr 20 150 255 --blob_thr 1000
        ;;
    "img_4")
        python main.py --img_path ../data/img_4.jpeg --low_thr 0 75 20 --high_thr 20 150 175 --blob_thr 1000
        ;;
    "img_5")
        python main.py --img_path ../data/img_5.jpg --low_thr 5 50 100 --high_thr 20 255 255 --blob_thr 10000
        ;;
    "img_6")
        python main.py --img_path ../data/img_6.jpg --low_thr 5 50 100 --high_thr 15 150 255 --blob_thr 2000
        ;;
    "img_7")
        python main.py --img_path ../data/img_7.jpg --low_thr 0 50 125 --high_thr 14 150 255 --blob_thr 15000
        ;;
esac