# Copyright (C) 2020 and later: Google, Inc.

"""This purpose of this script is to create source .txt files. For statistical
analysis on distribution of radicals and strokes, see Notebook script
'Radical-stroke_Index_Analysis.ipynb'.
Usage:
    """

import os
import pandas as pd
import random

def download_data():
    """Download Unihan.zip and unzip.

    Returns:
        data_path: Str, path to Unihan_RadicalStrokeCounts data.
    """
    # Download Unihan meta data for radical-stroke analysis
    os.system(' mkdir Unihan')
    os.system(' curl -O http://unicode.org/Public/UCD/latest/ucd/Unihan.zip')
    os.system(' apt-get -y install unzip')
    os.system(' unzip Unihan.zip -d Unihan/')
    os.system(' rm Unihan.zip')

    data_path = 'Unihan/Unihan_RadicalStrokeCounts.txt'
    assert(os.path.isfile(data_path))

    return data_path


def data_cleansing_and_preperation(data_path):
    """Clean and prepare data read in from data path.

    Kangxi radicals is a system of categorizing all Chinese characters based on
    one of their components. For example the radical for '冰' (ice),
    '冻' (freeze) and '凉' (cold) is '冫'(ice). The radical indicates the partial
    visual features of the character. See
    https://en.wikipedia.org/wiki/Kangxi_radical for more details.

    Strokes are the calligraphic strokes needed to write the Chinese characters
    in regular script used in East Asian calligraphy (used in China, Japan and
    Korea). The number of strokes usually indicates the complexity of the
    character. For example '一' has one stroke, '牛' has 4 strokes, '晴' has 12
    strokes and '撤' has 15 strokes. See
    https://en.wikipedia.org/wiki/Stroke_(CJK_character) for details.

    In this function, we do the following for each individual character in
    Unihan data:
        1. Get the radical number
        2. Get actual radical character
        3. Get stroke count for the radical
        4. Get stroke count for the rest of the character
        5. Get total stroke count

    Args:
        data_path: Str, path to Unihan_RadicalStrokeCounts data.

    Returns:
        df: pandas.DataFrame, with entries described as above.
    """
    # Mapping from radical number to radical string
    # Example:
    #   1 -> '⼀'
    #   2 -> '⼁'
    #   3 -> '⼂'
    #   ...
    # Radicals in Unicode range from '\u2f00' to '\u2fd6'
    # See https://en.wikipedia.org/wiki/List_of_radicals_in_Unicode
    number_to_radical = {(i + 1): chr(0x2F00 + i) for i in range(214)}

    # Get mapping from radical number to number of strokes in the radical
    # Radical number i has number_to_stroke_count[i] strokes
    # Example:
    #   1 ('⼀') -> 1
    #   ...
    #   8 ('亠') -> 2
    #   ...
    #   41 ('寸') -> 3
    #   ...
    radical_numbers = [i + 1 for i in range(214)]
    # For radical number i, the stroke count is stroke_counts[i-1]
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
    with open(data_path) as f_in:
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

                # Include all characters in CJK Unified Ideographs (4E00–9FFF)
                # and CJK Unified Ideographs Extension A (3400–4DBF). Avoid
                # all code point larger than 0x20000 (start of extension B) for
                # better font compatibility.
                if ord(character) <= 0x20000:
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

    return df


def add_char_by_radical(df, dataset, repeat=1):
    """For each radical, add oen character that contains that radical into
    the dataset. Repeat this process 'repeat' times.

    Args:
        df: pandas.DataFrame, with 'code_point', 'character', 'radical_number',
            'radical', 'radical_strokes', 'remaining_strokes' as columns.
        dataset: Set, a set of characters to include.
        repeat: Int, the number of time to repeat this process.
    """
    for j in range(repeat):
        for i in range(214):
            chars = df.loc[df['radical_number'] == i + 1]['character']\
                .tolist()
            char = random.choice(chars)
            dataset.add(char)


def add_char_by_stroke(df, dataset, repeat=1):
    """For each stroke count, add oen character that contains that radical into
    the dataset. Repeat this process 'repeat' times.

    Args:
        df: pandas.DataFrame, with 'code_point', 'character', 'radical_number',
            'radical', 'radical_strokes', 'remaining_strokes' as columns.
        dataset: Set, a set of characters to include.
        repeat: Int, the number of time to repeat this process.
    """
    stroke_list = df['total_strokes'].tolist()
    stroke_unique = (set(stroke_list))
    for j in range(repeat):
        for stroke_num in stroke_unique:
            chars = df.loc[df['total_strokes'] == stroke_num]['character']\
                .tolist()
            char = random.choice(chars)
            dataset.add(char)

def add_char_up_to(df, dataset, n=1000):
    """Randomly select character from df to add to dataset. Fill dataset until
    reach cardinality of n.

    Args:
        df: pandas.DataFrame, with 'code_point', 'character', 'radical_number',
            'radical', 'radical_strokes', 'remaining_strokes' as columns.
        dataset: Set, a set of characters to include.
        n: Int, will fill up dataset until reaches size of n.
    """
    chars = df['character'].tolist()
    while len(dataset) < n:
        dataset.add(random.choice(chars))

def output_file(data, filename):
    """Write dataset as code points into .txt file.

    Args:
        data: Set, dataset of characters to export.
        filename: Str, name of the output file.
    """
    with open(filename + '.txt', 'w+') as f_out:
        for char in data:
            f_out.write('U+' + str(hex(ord(char)))[2:] + '\n')


if __name__ == "__main__":
    data_path = download_data()
    df = data_cleansing_and_preperation(data_path)

    # We want to make sure outliercharacters with special features are
    # selected in the dataset. We want to make sure the dataset has a high
    # degree of diversity, in terms of radical and stroke count.

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

    # Add one character for each radical
    add_char_by_radical(df, dataset)

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

    # Fill dataset to 1000 randomly
    add_char_up_to(df, dataset, 1000)

    # Store dataset of size 1k in .txt file
    charset_1k = dataset.copy()
    output_file(charset_1k, 'charset_1k')


    # Select charset_2000
    add_char_by_radical(df, dataset, 2)
    add_char_by_stroke(df, dataset, 4)
    add_char_up_to(df, dataset, 2000)

    # Store dataset of size 2k in .txt file
    charset_2k = dataset.copy()
    output_file(charset_2k, 'charset_2k')


    # Select charset_4k
    add_char_by_radical(df, dataset, 4)
    add_char_by_stroke(df, dataset, 8)
    add_char_up_to(df, dataset, 4000)

    # Store dataset of size 4k in .txt file
    charset_4k = dataset.copy()
    output_file(charset_4k, 'charset_4k')

    # Generate randset_x
    dataset = set()
    add_char_up_to(df, dataset, 1000)
    randset_1k = dataset.copy()
    output_file(randset_1k, 'randset_1k')

    add_char_up_to(df, dataset, 2000)
    randset_2k = dataset.copy()
    output_file(randset_2k, 'randset_2k')

    add_char_up_to(df, dataset, 4000)
    randset_4k = dataset.copy()
    output_file(randset_4k, 'randset_4k')


    # Als store full dataset
    output_file(df['character'], 'full_dataset')




