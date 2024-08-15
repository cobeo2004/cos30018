import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import argparse
import os
from shutil import copyfile, move
from pathlib import Path

# https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick2_ochl, volume_overlay


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def removeOutput(finput):
    if(Path(finput)).is_file():
        os.remove(finput)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='a csv file of stock data', required=True)
    parser.add_argument('-l', '--seq_len',
                        help='num of sequence length', default=20)
    parser.add_argument('-lf', '--label_file',
                        help='a label_file')
    parser.add_argument('-d', '--dimension',
                        help='a dimension value', type=int, default=48)
    parser.add_argument('-t', '--dataset_type',
                        help='training or testing datasets')
    parser.add_argument('-m', '--mode',
                        help='mode of preprocessing data', required=True)
    args = parser.parse_args()
    if args.mode == 'ohlc2cs':
        ohlc2cs(args.input, args.seq_len, args.dataset_type, args.dimension)
    if args.mode == 'createLabel':
        createLabel(args.input, args.seq_len)
    if args.mode == 'img2dt':
        image2dataset(args.input, args.label_file)
    if args.mode == 'countImg':
        countImage(args.input)


def image2dataset(input, label_file):
    label_dict = {}
    with open(label_file) as f:
        for line in f:
            (key, val) = line.split(',')
            label_dict[key] = val.rstrip()
    path = "{}/{}".format(os.getcwd(), input)
    for filename in os.listdir(path):
        if filename is not '':
            for k, v in label_dict.items():
                splitname = filename.split("_")
                f, e = os.path.splitext(filename)
                newname = "{}_{}".format(splitname[0], splitname[1])
                if newname == k:
                    new_name = "{}{}.png".format(v, f)
                    os.rename("{}/{}".format(path, filename),
                              "{}/{}".format(path, new_name))
                    break

    folders = ['1', '0']
    for folder in folders:
        if not os.path.exists("{}/classes/{}".format(path, folder)):
            os.makedirs("{}/classes/{}".format(path, folder))

    for filename in os.listdir(path):
        if filename is not '':
            if filename[:1] == "1":
                move("{}/{}".format(path, filename),
                     "{}/classes/1/{}".format(path, filename))
            elif filename[:1] == "0":
                move("{}/{}".format(path, filename),
                     "{}/classes/0/{}".format(path, filename))


def createLabel(fname, seq_len):
    print("Creating label . . .")
    filename = fname.split('/')
    removeOutput("{}_label_{}.txt".format(filename[1][:-4], seq_len))

    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    df.fillna(0)

    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    for i in range(0, len(df)):
        c = df.ix[i:i + int(seq_len), :]
        starting = 0
        endvalue = 0
        label = ""

        if len(c) == int(seq_len) + 1:
            for idx, val in enumerate(c['Close']):
                if idx == 0:
                    starting = float(val)
                if idx == len(c) - 1:
                    endvalue = float(val)
            if endvalue > starting:
                label = 1
            else:
                label = 0
            with open("{}_label_{}.txt".format(filename[1][:-4], seq_len), 'a') as the_file:
                the_file.write("{}-{},{}".format(filename[1][:-4], i, label))
                the_file.write("\n")
    print("Create label finished.")


def countImage(input):
    num_file = sum([len(files) for r, d, files in os.walk(input)])
    num_dir = sum([len(d) for r, d, files in os.walk(input)])
    print("num of files : {}\nnum of dir : {}".format(num_file, num_dir))


def ohlc2cs(fname, seq_len, dataset_type, dimension):
    print("Converting olhc to candlestick")
    symbol = fname.split('_')[0]
    symbol = symbol.split('/')[1]
    print(symbol)
    path = "{}".format(os.getcwd())
    # print(path)
    if not os.path.exists("{}/dataset/{}_{}/{}/{}".format(path, seq_len, dimension, symbol, dataset_type)):
        os.makedirs("{}/dataset/{}_{}/{}/{}".format(path,
                                                    seq_len, dimension, symbol, dataset_type))

    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    df.fillna(0)
    plt.style.use('dark_background')
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    for i in range(0, len(df)):
        # normal length - begin
        # candlestick ohlc normal
        c = df.ix[i:i + int(seq_len) - 1, :]
        # ohlc+volume
        useVolume = True
        if len(c) == int(seq_len):
            my_dpi = 96
            fig = plt.figure(figsize=(dimension / my_dpi,
                                      dimension / my_dpi), dpi=my_dpi)
            ax1 = fig.add_subplot(1, 1, 1)
            candlestick2_ochl(ax1, c['Open'], c['Close'], c['High'],
                              c['Low'], width=1,
                              colorup='#77d879', colordown='#db3f3f')
            ax1.grid(False)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis('off')

            # create the second axis for the volume bar-plot
            # Add a seconds axis for the volume overlay
            if useVolume:
                ax2 = ax1.twinx()
                # Plot the volume overlay
                bc = volume_overlay(ax2, c['Open'], c['Close'], c['Volume'],
                                    colorup='#77d879', colordown='#db3f3f', alpha=0.5, width=1)
                ax2.add_collection(bc)
                ax2.grid(False)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax2.axis('off')
            pngfile = 'dataset/{}_{}/{}/{}/{}-{}_combination.png'.format(
                seq_len, dimension, symbol, dataset_type, fname[11:-4], i)
            fig.savefig(pngfile,  pad_inches=0, transparent=False)
            plt.close(fig)

    print("Converting olhc to candlestik finished.")


if __name__ == '__main__':
    main()
