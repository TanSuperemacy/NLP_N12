
---
license: unknown
language: 
- vie
pretty_name: Uit Vsmec
task_categories: 
- emotion-classification
tags: 
- emotion-classification
---

This dataset consists of Vietnamese Facebook comments that were manually annotated for sentiment.
There are seven possible emotion labels: enjoyment, sadness, fear, anger, disgust, surprise or other (for comments with no or neutral emotions).
Two rounds of manual annotations were done to train annotators with tagging and editing guidelines.
Annotation was performed until inter-annotator agreement reached at least 80%.


## Languages

vie

## Supported Tasks

Emotion Classification

## Dataset Usage
### Using `datasets` library
```
from datasets import load_dataset
dset = datasets.load_dataset("SEACrowd/uit_vsmec", trust_remote_code=True)
```
### Using `seacrowd` library
```import seacrowd as sc
# Load the dataset using the default config
dset = sc.load_dataset("uit_vsmec", schema="seacrowd")
# Check all available subsets (config names) of the dataset
print(sc.available_config_names("uit_vsmec"))
# Load the dataset using a specific config
dset = sc.load_dataset_by_config_name(config_name="<config_name>")
```

More details on how to load the `seacrowd` library can be found [here](https://github.com/SEACrowd/seacrowd-datahub?tab=readme-ov-file#how-to-use).


## Dataset Homepage

[https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu](https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu)

## Dataset Version

Source: 1.0.0. SEACrowd: 2024.06.20.

## Dataset License

Unknown (unknown)

## Citation

If you are using the **Uit Vsmec** dataloader in your work, please cite the following:
```
@inproceedings{ho2020emotion,
  title={Emotion recognition for vietnamese social media text},
  author={Ho, Vong Anh and Nguyen, Duong Huynh-Cong and Nguyen, Danh Hoang and Pham, Linh Thi-Van and Nguyen, Duc-Vu and Nguyen, Kiet Van and Nguyen, Ngan Luu-Thuy},
  booktitle={Computational Linguistics: 16th International Conference of the Pacific Association for Computational Linguistics, PACLING 2019, Hanoi, Vietnam, October 11--13, 2019, Revised Selected Papers 16},
  pages={319--333},
  year={2020},
  organization={Springer}
}


@article{lovenia2024seacrowd,
    title={SEACrowd: A Multilingual Multimodal Data Hub and Benchmark Suite for Southeast Asian Languages}, 
    author={Holy Lovenia and Rahmad Mahendra and Salsabil Maulana Akbar and Lester James V. Miranda and Jennifer Santoso and Elyanah Aco and Akhdan Fadhilah and Jonibek Mansurov and Joseph Marvin Imperial and Onno P. Kampman and Joel Ruben Antony Moniz and Muhammad Ravi Shulthan Habibi and Frederikus Hudi and Railey Montalan and Ryan Ignatius and Joanito Agili Lopo and William Nixon and Börje F. Karlsson and James Jaya and Ryandito Diandaru and Yuze Gao and Patrick Amadeus and Bin Wang and Jan Christian Blaise Cruz and Chenxi Whitehouse and Ivan Halim Parmonangan and Maria Khelli and Wenyu Zhang and Lucky Susanto and Reynard Adha Ryanda and Sonny Lazuardi Hermawan and Dan John Velasco and Muhammad Dehan Al Kautsar and Willy Fitra Hendria and Yasmin Moslem and Noah Flynn and Muhammad Farid Adilazuarda and Haochen Li and Johanes Lee and R. Damanhuri and Shuo Sun and Muhammad Reza Qorib and Amirbek Djanibekov and Wei Qi Leong and Quyet V. Do and Niklas Muennighoff and Tanrada Pansuwan and Ilham Firdausi Putra and Yan Xu and Ngee Chia Tai and Ayu Purwarianti and Sebastian Ruder and William Tjhi and Peerat Limkonchotiwat and Alham Fikri Aji and Sedrick Keh and Genta Indra Winata and Ruochen Zhang and Fajri Koto and Zheng-Xin Yong and Samuel Cahyawijaya},
    year={2024},
    eprint={2406.10118},
    journal={arXiv preprint arXiv: 2406.10118}
}

```