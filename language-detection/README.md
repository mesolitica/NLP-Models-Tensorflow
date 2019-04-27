## How-to

1. You need to download and process dataset first,
```bash
wget http://downloads.tatoeba.org/exports/sentences.tar.bz2
bunzip2 sentences.tar.bz2
tar xvf sentences.tar
```

2. Change to csv,
```bash
awk -F"\t" '{print"__label__"$2" "$3}' < sentences.csv | shuf > all.txt
```

3. Run any notebook using Jupyter Notebook.
