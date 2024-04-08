<<<<<<< HEAD
---
annotations_creators:
- no-annotation
language_creators:
- found
language:
- cs
- de
- en
- fr
- hi
- ru
license:
- unknown
multilinguality:
- translation
size_categories:
- 10M<n<100M
source_datasets:
- extended|europarl_bilingual
- extended|giga_fren
- extended|news_commentary
- extended|un_multi
- extended|hind_encorp
task_categories:
- translation
task_ids: []
paperswithcode_id: wmt-2014
pretty_name: WMT14
dataset_info:
- config_name: cs-en
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - cs
        - en
  splits:
  - name: train
    num_bytes: 280992026
    num_examples: 953621
  - name: validation
    num_bytes: 702465
    num_examples: 3000
  - name: test
    num_bytes: 757809
    num_examples: 3003
  download_size: 168878237
  dataset_size: 282452300
- config_name: de-en
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - de
        - en
  splits:
  - name: train
    num_bytes: 1358406800
    num_examples: 4508785
  - name: validation
    num_bytes: 736407
    num_examples: 3000
  - name: test
    num_bytes: 777326
    num_examples: 3003
  download_size: 818467512
  dataset_size: 1359920533
- config_name: fr-en
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - fr
        - en
  splits:
  - name: train
    num_bytes: 14752522252
    num_examples: 40836715
  - name: validation
    num_bytes: 744439
    num_examples: 3000
  - name: test
    num_bytes: 838849
    num_examples: 3003
  download_size: 7777527744
  dataset_size: 14754105540
- config_name: hi-en
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - hi
        - en
  splits:
  - name: train
    num_bytes: 1936003
    num_examples: 32863
  - name: validation
    num_bytes: 181457
    num_examples: 520
  - name: test
    num_bytes: 1075008
    num_examples: 2507
  download_size: 1583004
  dataset_size: 3192468
- config_name: ru-en
  features:
  - name: translation
    dtype:
      translation:
        languages:
        - ru
        - en
  splits:
  - name: train
    num_bytes: 433209078
    num_examples: 1486965
  - name: validation
    num_bytes: 977938
    num_examples: 3000
  - name: test
    num_bytes: 1087738
    num_examples: 3003
  download_size: 223537244
  dataset_size: 435274754
configs:
- config_name: cs-en
  data_files:
  - split: train
    path: cs-en/train-*
  - split: validation
    path: cs-en/validation-*
  - split: test
    path: cs-en/test-*
- config_name: de-en
  data_files:
  - split: train
    path: de-en/train-*
  - split: validation
    path: de-en/validation-*
  - split: test
    path: de-en/test-*
- config_name: fr-en
  data_files:
  - split: train
    path: fr-en/train-*
  - split: validation
    path: fr-en/validation-*
  - split: test
    path: fr-en/test-*
- config_name: hi-en
  data_files:
  - split: train
    path: hi-en/train-*
  - split: validation
    path: hi-en/validation-*
  - split: test
    path: hi-en/test-*
- config_name: ru-en
  data_files:
  - split: train
    path: ru-en/train-*
  - split: validation
    path: ru-en/validation-*
  - split: test
    path: ru-en/test-*
---

# Dataset Card for "wmt14"

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [http://www.statmt.org/wmt14/translation-task.html](http://www.statmt.org/wmt14/translation-task.html)
- **Repository:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Paper:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Point of Contact:** [More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
- **Size of downloaded dataset files:** 1.70 GB
- **Size of the generated dataset:** 282.95 MB
- **Total amount of disk used:** 1.98 GB

### Dataset Summary

<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
  <p><b>Warning:</b> There are issues with the Common Crawl corpus data (<a href="https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz">training-parallel-commoncrawl.tgz</a>):</p>
  <ul>
    <li>Non-English files contain many English sentences.</li>
    <li>Their "parallel" sentences in English are not aligned: they are uncorrelated with their counterpart.</li>
  </ul>
  <p>We have contacted the WMT organizers, and in response, they have indicated that they do not have plans to update the Common Crawl corpus data. Their rationale pertains to the expectation that such data has been superseded, primarily by CCMatrix, and to some extent, by ParaCrawl datasets.</p>
</div>

Translation dataset based on the data from statmt.org.

Versions exist for different years using a combination of data
sources. The base `wmt` allows you to create a custom dataset by choosing
your own data/language pair. This can be done as follows:

```python
from datasets import inspect_dataset, load_dataset_builder

inspect_dataset("wmt14", "path/to/scripts")
builder = load_dataset_builder(
    "path/to/scripts/wmt_utils.py",
    language_pair=("fr", "de"),
    subsets={
        datasets.Split.TRAIN: ["commoncrawl_frde"],
        datasets.Split.VALIDATION: ["euelections_dev2019"],
    },
)

# Standard version
builder.download_and_prepare()
ds = builder.as_dataset()

# Streamable version
ds = builder.as_streaming_dataset()
```

### Supported Tasks and Leaderboards

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Languages

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Dataset Structure

### Data Instances

#### cs-en

- **Size of downloaded dataset files:** 1.70 GB
- **Size of the generated dataset:** 282.95 MB
- **Total amount of disk used:** 1.98 GB

An example of 'train' looks as follows.
```

```

### Data Fields

The data fields are the same among all splits.

#### cs-en
- `translation`: a multilingual `string` variable, with possible languages including `cs`, `en`.

### Data Splits

|name |train |validation|test|
|-----|-----:|---------:|---:|
|cs-en|953621|      3000|3003|

## Dataset Creation

### Curation Rationale

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the source language producers?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Annotations

#### Annotation process

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

#### Who are the annotators?

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Personal and Sensitive Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Discussion of Biases

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Other Known Limitations

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

## Additional Information

### Dataset Curators

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Licensing Information

[More Information Needed](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)

### Citation Information

```

@InProceedings{bojar-EtAl:2014:W14-33,
  author    = {Bojar, Ondrej  and  Buck, Christian  and  Federmann, Christian  and  Haddow, Barry  and  Koehn, Philipp  and  Leveling, Johannes  and  Monz, Christof  and  Pecina, Pavel  and  Post, Matt  and  Saint-Amand, Herve  and  Soricut, Radu  and  Specia, Lucia  and  Tamchyna, Ale
{s}},
  title     = {Findings of the 2014 Workshop on Statistical Machine Translation},
  booktitle = {Proceedings of the Ninth Workshop on Statistical Machine Translation},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {12--58},
  url       = {http://www.aclweb.org/anthology/W/W14/W14-3302}
}

```


### Contributions

Thanks to [@thomwolf](https://github.com/thomwolf), [@patrickvonplaten](https://github.com/patrickvonplaten) for adding this dataset.
=======
# pref_opt

.....
>>>>>>> bb4ceb3b980db5e3cca25f9c82c43665e43b08bb
