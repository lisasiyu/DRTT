# DRTT
Code for NAACL 2022 ["Generating Authentic Adversarial Examples beyond Meaning-preserving with Doubly Round-trip Translation"](https://aclanthology.org/2022.naacl-main.316.pdf)
## STEP 1 Training M-MLM and T-MLM
```
sh ./myscript/train_mlm.sh
sh ./myscript/train_tlm.sh
```
## STEP 2 Training forward and backward baseline models
```
sh ./myscript/train_baseline.sh
```

## STEP 3 Generating adversarial examples
```
sh ./myscript/multi_data_aug.sh
```

## STEP 4 Filtering with our definition
```
sh ./myscript/filter.sh
```
## STEP 5 Training DRTT model
```
sh ./myscript/train_chen.sh
```

## STEP 6 Testing on noisy testset
```
sh ./myscript/test_noisy.sh
```
## Citation
Please cite the following paper if you found the resources in this repository useful.
```
@inproceedings{lai-etal-2022-generating,
    title = "Generating Authentic Adversarial Examples beyond Meaning-preserving with Doubly Round-trip Translation",
    author = "Lai, Siyu  and
      Yang, Zhen  and
      Meng, Fandong  and
      Zhang, Xue  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.316",
    doi = "10.18653/v1/2022.naacl-main.316",
    pages = "4256--4266",
}

```
