# 🔍 WIKIDUMP SEARCH

It is an offline utility/tool made for searching 'keywords' in [Wikipedia Archive](https://dumps.wikimedia.org/enwiki/) instead of using any online WikipediaAPI. 

---

# 🎯 BENEFITS 

- when you need to search for many 'keywords' in Wikipedia. WikipediaAPI such as [Wikipedia](https://pypi.org/project/wikipedia/) may slow down after few dozens of calls.
- if your internet connection is not fast, then this is beneficial as it is an offline search.
- uses very minimal onboard resource. 

---

# 🛠️ REQUIREMENTS

- tested on Python 3.11
- [Wikipedia](https://pypi.org/project/wikipedia/)
- [FuzzyWuzzy](https://pypi.org/project/fuzzywuzzy/)
- [Beautifulsoup](https://pypi.org/project/beautifulsoup4/)
- [tdqm](https://pypi.org/project/tqdm/)
- [joblib](https://pypi.org/project/joblib/)
- atleast 25 GB free storage space

or you can install using `pip install -r "./requirements.txt" `

Also, you need to download one image/backup from this [wiki-archive page](https://dumps.wikimedia.org/enwiki/) 

---

# ⚙️ SETUP

Download 
- `enwiki-{data}-pages-articles-multistream.xml.bz2` (~23 GB) 
- `enwiki-{date}-pages-articles-multistream-index.txt.bz2` (~250 MB)  
  - Extract this file. It will contain `enwiki-{date}-pages-articles-multistream-index.txt` (~1.2 GB) 

These file's filepaths will be required when initializing thhe offline wiki class

---

# 📝 EXAMPLE

See [testing.ipynb](./testing.ipynb)
