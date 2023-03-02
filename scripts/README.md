# Scripts

This README provides instruction on how to use the scripts in this directory for
data pre-processing and FastLAS learning tasks.

## Data pre-processing

You will need the raw pkl files that we converted from Concept Bottleneck 
Models' paper [1]'s raw pkl to our `CUBRawDataItem` (class in `common.py`). This
conversion only makes the raw data point in CBM's pkl into more readable format.
Download link: [raw_pkl.tar.gz](https://drive.google.com/file/d/1wbSbkKgVN-i8ZBjiRyRf93zTKATeU0Is/view?usp=share_link)

Once you have the raw pkl files, you can run `data_preprocess.sh` (we provide an
example `data_process_example`). The key args are `-nc` (number of classes), `-t`
(min_class_threshold). You can follow the below table to generate the pre-processed
subsets/dataset that we use in our paper.

| Dataset | Attributes (after pre-processing) | Classes | min_class_threshold | Train | Val  | Test | Total |
|---------|-----------------------------------|---------|---------------------|-------|------|------|-------|
| CUB-3   | 34                                | 3       | 1                   | 71    | 19   | 88   | 178   |
| CUB-10  | 41                                | 10      | 2                   | 240   | 60   | 243  | 543   |
| CUB-15  | 40                                | 15      | 3                   | 363   | 87   | 387  | 837   |
| CUB-20  | 48                                | 20      | 3                   | 486   | 114  | 515  | 1115  |
| CUB-25  | 50                                | 25      | 3                   | 612   | 138  | 652  | 1402  |
| CUB-50  | 61                                | 50      | 5                   | 1183  | 317  | 1389 | 2889  |
| CUB-100 | 82                                | 100     | 7                   | 2399  | 601  | 2864 | 5864  |
| CUB-200 | 112                               | 200     | 10                  | 4796  | 1198 | 5794 | 11788 |

## FastLAS learning tasks

You will need to install [FastLAS](https://github.com/spike-imperial/FastLAS/releases)
and clingo first.
We provide a `las_prep_example` that calls `las_gen.py` to generate mode biases
and learning examples for a LAS task. The script takes pre-processed dataset and
will dump the `.las` files into a directory (you need to create one first before
pass it in as an arg). For example, after running `las_prep.sh` on CUB-3 subset,
the output background file (that has mode biases, but no actual background
knowledge) is `bk_c3.las` and the examples file is `example_c3.las`, to learn
with FastLAS, run:

```
FastLAS --opl bk_c3.las example_c3.las
```

## Reference

[1] [Pang Wei Koh, Thao Nguyen, Yew Siang Tang et al., Concept Bottleneck Models](http://proceedings.mlr.press/v119/koh20a.html)
