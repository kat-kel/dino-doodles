# dino-doodles

### Get started

1. Clone this repository and change to that directory.

2. Create and activate a virtual environment for python 3.7.

3. Install the project's dependencies in that virtual environment.

```
pip install -r requirements.txt
```

4. Work with the example data or install your own CSV (with column "Proposition") as `private.csv` in the folder `data/`.

### Explore functions

- Run the TF-IDF analysis. Customize the command with a maximum document frequency (`--max_df`, float), a minimum document frequency (`--min_df`, integer), a number of clusters (`--clusters`, integer), and the number of terms that compose a cluster (`--members`, integer).

```
python run.py --max_df 0.4 --min_df 5 --clusters 5 --members 8
```
