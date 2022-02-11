# Joint Abductive and Inductive Neural Logical Reasoning

# Requirements
* python == 3.8.5
* torch == 1.8.1
* numpy == 1.19.2
* pandas == 1.0.1
* tqdm == 4.61.0


# $\textbf{ABIN}$ for ABductive and INductive NLR
To reproduce the main results, run the following commands:

    nohup python ./code/ABIN.py --dataset YAGO4 >../log/your_log_file 2>&1 &
    nohup python ./code/ABIN.py --dataset DBpedia >../log/your_log_file 2>&1 &


# AI-NLR Datasets

## Yago4
### Use pre-processed datasets
Default. See files in:

    ./data/YAGO4/input/

### Build from source
Download and unzip the raw data:

$\mathcal{T}$: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-class.nt.gz

$\mathcal{A}_{ee}$: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-facts.nt.gz

$\mathcal{A}_{ec}$: https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-full-types.nt.gz and https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-simple-types.nt.gz



Run all cells in:

    ./code/ppc_YAGO4/raw2mid.ipynb
    ./code/ppc_YAGO4/ppc.ipynb


## DBpedia
### Use pre-processed datasets
Default. See files in:

    ./data/DBpedia/input/

### Build from source
Download and unzip the raw data:

$\mathcal{T}$: http://downloads.dbpedia.org/2016-10/dbpedia_2016-10.nt

$\mathcal{A}_{ee}$: http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_objects_wkd_uris_en.ttl.bz2

$\mathcal{A}_{ec}$: http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_transitive_wkd_uris_en.ttl.bz2

Run all cells in:

    ./code/ppc_DBpedia/raw2mid.ipynb
    ./code/ppc_DBpedia/ppc.ipynb