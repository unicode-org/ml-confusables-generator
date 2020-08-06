# Copyright (C) 2020 and later: Google, Inc.

"""This purpose of this script is to create source .txt files. For statistical
analysis on distribution of radicals and strokes, see Notebook script
'Radical-stroke_Index_Analysis.ipynb'.
Usage:
    """

import os
import pandas as pd
import random


def main():
    # Download Unihan meta data for radical-stroke analysis
    os.system(' mkdir Unihan')
    os.system(' curl -O http://unicode.org/Public/UCD/latest/ucd/Unihan.zip')
    os.system(' apt-get -y install unzip')
    os.system(' unzip Unihan.zip -d Unihan/')
    os.system(' rm Unihan.zip')

    # Mapping from radical number to radical string
    # Radicals in Unicode range from '\u2f00' to '\u2fd6'
    # See https://en.wikipedia.org/wiki/List_of_radicals_in_Unicode
    number_to_radical = {(i + 1): chr(int('0x2F00', 16) + i)
                         for i in range(214)}

    # Get mapping from radical number to number of strokes in the radical
    # Radical number i has number_to_stroke_count[i] strokes
    radical_numbers = [i + 1 for i in range(214)]
    stroke_counts = [1] * 6 + [2] * 23 + [3] * 31 + [4] * 34 + [5] * 23 + \
                    [6] * 29 + [7] * 20 + [8] * 9 + [9] * 11 + [10] * 8 + \
                    [11] * 6 + [12] * 4 + [13] * 4 + [14] * 2 + [15] * 1 + \
                    [16] * 2 + [17] * 1
    number_to_stroke_count = dict(zip(radical_numbers, stroke_counts))

    # Each entry in codepoint_radical_stroke_list is:
    # (code_point, character, radical_number, radical, radical_stroke_count,
    # remaining_stroke_count)
    codepoint_radical_stroke_list = []

    # Create intermediate txt file that includes meta data
    with open('Unihan/Unihan_RadicalStrokeCounts.txt') as f_in:
        for line in f_in:
            if line[0] == '#' or len(line) <= 1:
                # Skip if the current line is not what we are looking for
                continue
            elif line.split('\t')[1] == 'kRSKangXi':
                # Read in fields separated by tabs
                # Example for each line:
                # U+3A16	kRSKangXi	64.9
                fields = line.split('\t')

                codepoint, radical, stroke = fields[0], \
                                             fields[2].split('.')[0], \
                                             fields[2].split('.')[-1][:-1]
                # Get character for each code point
                character = chr(int('0x' + codepoint[2:], 16))
                # Get radical index
                radical_index = int(radical)
                # Get radical character for each radical index
                radical = number_to_radical[radical_index]
                # Get stroke count for each radical
                radical_stroke_count = number_to_stroke_count[radical_index]
                # Get remaining stroke count
                remaining_stroke_count = int(stroke)
                if ord(character) <= 131072:
                    entry = (codepoint, character, radical_index, radical,
                             radical_stroke_count, remaining_stroke_count)
                    codepoint_radical_stroke_list.append(entry)

    # Build pandas DataFrame for data analysis
    df = pd.DataFrame(codepoint_radical_stroke_list,
                      columns = ['code_point', 'character', 'radical_number',
                                 'radical', 'radical_strokes',
                                 'remaining_strokes'])
    df['total_strokes'] = df['radical_strokes'] + df['remaining_strokes']

    # Check if all code points are unique in the dataset
    code_points = df['code_point'].tolist()
    assert (len(code_points) == len(set(code_points)))

    # Get count for each radical
    radical_num_list = df['radical_number'].tolist()
    radical_count = dict()
    for i in range(214):
        radical_count[i+1] = radical_num_list.count(i+1)

    # We want to make sure **outlier** characters with special features are
    # selected in the dataset. We want to make sure the dataset has a high
    # degree of **diversity**.

    # Initialize dataset
    dataset = set()

    # Include all individual radicals in the dataset
    radical_set = set(df['radical'])
    dataset = dataset.union(radical_set)


    # ### Select character with basic and compound strokes
    # See https://en.wikipedia.org/wiki/Stroke_(CJK_character)
    # Add group of characters that includes all basic and compound strokes
    basic_stroke_chars = ['二', '孑', '了', '又', '口', '勺', '计', '凹', '殳',
                          '飞', '艺', '凸', '及', '队', '乃', '中', '水', '永',
                          '以', '山', '亡', '肅', '己', '亞', '丂', '乂', '人',
                          '乄', '公', '巡', '火', '尺', '之', '弋', '必', '心',
                          '狐']
    dataset = set.union(dataset, set(basic_stroke_chars))


    # Select one other character for each radical
    random.seed(2066)

    # Define function that add character to dataset based on radical
    def add_char_by_radical(df, dataset, repeat=1):
        for j in range(repeat):
            for i in range(214):
                chars = df.loc[df['radical_number'] == i + 1]['character']\
                    .tolist()
                char = random.choice(chars)
                dataset.add(char)

    # Add one character for each radical
    add_char_by_radical(df, dataset)


    # Define function that add character to dataset based n total number of
    # strokes
    def add_char_by_stroke(df, dataset, repeat=1):
        stroke_list = df['total_strokes'].tolist()
        stroke_unique = (set(stroke_list))
        for j in range(repeat):
            for stroke_num in stroke_unique:
                chars = df.loc[df['total_strokes'] == stroke_num]['character']\
                    .tolist()
                char = random.choice(chars)
                dataset.add(char)

    # Add 2 characters for each storke number
    add_char_by_stroke(df, dataset, 2)

    # Select simple characters (total_strokes <= 4)
    chars = df.loc[df['total_strokes'] <= 4]['character'].tolist()
    chars = random.choices(chars, k=80)
    dataset = set.union(dataset, set(chars))

    # Select complicated characters (total_strokes >= 27)
    chars = df.loc[df['total_strokes'] >= 27]['character'].tolist()
    chars = random.choices(chars, k=80)
    dataset = set.union(dataset, set(chars))

    # Define function to randomly fill dataset up to n characters
    def add_char_up_to(df, dataset, n=1000):
        chars = df['character'].tolist()
        while len(dataset) < n:
            dataset.add(random.choice(chars))

    # Fill dataset to 1000 randomly
    add_char_up_to(df, dataset, 1000)

    # Name current dataset charset_1k for later use
    charset_1k = dataset.copy()


    # Select charset_2000
    add_char_by_radical(df, dataset, 2)
    add_char_by_stroke(df, dataset, 4)
    add_char_up_to(df, dataset, 2000)

    charset_2k = dataset.copy()


    # Select charset_4000
    add_char_by_radical(df, dataset, 4)
    add_char_by_stroke(df, dataset, 8)
    add_char_up_to(df, dataset, 4000)

    charset_4k = dataset.copy()


    # Generate randset_x
    dataset = set()
    add_char_up_to(df, dataset, 1000)
    randset_1k = dataset.copy()
    add_char_up_to(df, dataset, 2000)
    randset_2k = dataset.copy()
    add_char_up_to(df, dataset, 4000)
    randset_4k = dataset.copy()

    # Define function for storing dataset to txt file
    def output_file(data, filename):
        with open(filename + '.txt', 'w+') as f_out:
            for char in data:
                f_out.write('U+' + str(hex(ord(char)))[2:] + '\n')

    # Store datasets
    datasets = [charset_1k, charset_2k, charset_4k, randset_1k, randset_2k,
                randset_4k]
    data_names = ['charset_1k', 'charset_2k', 'charset_4k', 'randset_1k',
                  'randset_2k', 'randset_4k']
    for i in range(6):
        output_file(datasets[i], data_names[i])


    # Als store full dataset
    output_file(df['character'], 'full_dataset')

if __name__ == "__main__":
    main()




