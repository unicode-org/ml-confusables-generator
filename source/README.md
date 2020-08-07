# Source File Generation

This directory is used for dataset source file generation. Source files contains
Unicode code points in Han Script selected to be in the dataset. Statistical 
analysis on radical and stroke distributions were done to make sure of the
diversity of the visual features in `charset_*.txt`. Meta data about characters'
radical stroke index is gathered from Unihan database
([link](https://unicode.org/charts/unihan.html)).

## Files
- Generation scripts:
    - `Radical-stroke_Index_Analysis.ipynb`: Jupyter Notebook for radical-stroke
    analysis and dataset selection.
    - `Radical-stroke_Index_Analysis.py`: Produces the same result as Jupyter
    Notebook file.
- Source files:
    - charset_1k.txt: selected 1000 Unicode code points.
    - charset_2k.txt: selected 2000 Unicode code points.
    - charset_4k.txt: selected 4000 Unicode code points.
    - randset_1k.txt: randomly selected 1000 Unicode code points.
    - randset_2k.txt: randomly selected 2000 Unicode code points.
    - randset_4k.txt: randomly selected 4000 Unicode code points.
    - full_dataset.txt: full dataset containing 21028 code points.
    
## Usage
- Scripts:
    - `Radical-stroke_Index_Analysis.ipynb`: Open .ipynb file in Jupyter 
    Notebook, select tab Cell, then Run All. The entire script should finish 
    running in less than 1 minute. 
    - `generate_source_file.py`: Run `python generate_source_file.py`
- Source files:
All source files can be used by VisualGenerator object in vis_gen module. 
```python
vg = VisualGenerator()
vg.generate_dataset_from_file('source/charset_1k.txt', 
                              ['Bold','Medium','Regular'],
                              ['Default','None'])
```