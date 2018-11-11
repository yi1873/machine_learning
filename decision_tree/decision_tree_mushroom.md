Identifying poisonousmushrooms with rule learners
================
November 06, 2018

Information
===========

-   数据获取 <https://github.com/stedy/Machine-Learning-with-R-datasets>
-   R markdown整理 <https://github.com/yi1873/machine_learning>

Step 1. Prepare data
--------------------

-   读取数据

``` r
mushrooms <- read.csv("../data/mushrooms.csv", stringsAsFactors = TRUE)
str(mushrooms)
```

    ## 'data.frame':    8124 obs. of  23 variables:
    ##  $ type                    : Factor w/ 2 levels "e","p": 2 1 1 2 1 1 1 1 2 1 ...
    ##  $ cap_shape               : Factor w/ 6 levels "b","c","f","k",..: 6 6 1 6 6 6 1 1 6 1 ...
    ##  $ cap_surface             : Factor w/ 4 levels "f","g","s","y": 3 3 3 4 3 4 3 4 4 3 ...
    ##  $ cap_color               : Factor w/ 10 levels "b","c","e","g",..: 5 10 9 9 4 10 9 9 9 10 ...
    ##  $ bruises                 : Factor w/ 2 levels "f","t": 2 2 2 2 1 2 2 2 2 2 ...
    ##  $ odor                    : Factor w/ 9 levels "a","c","f","l",..: 7 1 4 7 6 1 1 4 7 1 ...
    ##  $ gill_attachment         : Factor w/ 2 levels "a","f": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ gill_spacing            : Factor w/ 2 levels "c","w": 1 1 1 1 2 1 1 1 1 1 ...
    ##  $ gill_size               : Factor w/ 2 levels "b","n": 2 1 1 2 1 1 1 1 2 1 ...
    ##  $ gill_color              : Factor w/ 12 levels "b","e","g","h",..: 5 5 6 6 5 6 3 6 8 3 ...
    ##  $ stalk_shape             : Factor w/ 2 levels "e","t": 1 1 1 1 2 1 1 1 1 1 ...
    ##  $ stalk_root              : Factor w/ 5 levels "?","b","c","e",..: 4 3 3 4 4 3 3 3 4 3 ...
    ##  $ stalk_surface_above_ring: Factor w/ 4 levels "f","k","s","y": 3 3 3 3 3 3 3 3 3 3 ...
    ##  $ stalk_surface_below_ring: Factor w/ 4 levels "f","k","s","y": 3 3 3 3 3 3 3 3 3 3 ...
    ##  $ stalk_color_above_ring  : Factor w/ 9 levels "b","c","e","g",..: 8 8 8 8 8 8 8 8 8 8 ...
    ##  $ stalk_color_below_ring  : Factor w/ 9 levels "b","c","e","g",..: 8 8 8 8 8 8 8 8 8 8 ...
    ##  $ veil_type               : Factor w/ 1 level "p": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ veil_color              : Factor w/ 4 levels "n","o","w","y": 3 3 3 3 3 3 3 3 3 3 ...
    ##  $ ring_number             : Factor w/ 3 levels "n","o","t": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ ring_type               : Factor w/ 5 levels "e","f","l","n",..: 5 5 5 5 1 5 5 5 5 5 ...
    ##  $ spore_print_color       : Factor w/ 9 levels "b","h","k","n",..: 3 4 4 3 4 3 3 4 3 3 ...
    ##  $ population              : Factor w/ 6 levels "a","c","n","s",..: 4 3 3 4 1 3 3 4 5 4 ...
    ##  $ habitat                 : Factor w/ 7 levels "d","g","l","m",..: 6 2 4 6 2 2 4 4 2 4 ...

``` r
mushrooms$veil_type <- NULL
table(mushrooms$type)
```

    ## 
    ##    e    p 
    ## 4208 3916

Step 2. Training a model on the data
------------------------------------

``` r
library(RWeka)

mushroom_1R <- OneR(type ~ ., data = mushrooms)
mushroom_1R
```

    ## odor:
    ##  a   -> e
    ##  c   -> p
    ##  f   -> p
    ##  l   -> e
    ##  m   -> p
    ##  n   -> e
    ##  p   -> p
    ##  s   -> p
    ##  y   -> p
    ## (8004/8124 instances correct)

Step 3. Evaluating model performance
------------------------------------

``` r
summary(mushroom_1R)
```

    ## 
    ## === Summary ===
    ## 
    ## Correctly Classified Instances        8004               98.5229 %
    ## Incorrectly Classified Instances       120                1.4771 %
    ## Kappa statistic                          0.9704
    ## Mean absolute error                      0.0148
    ## Root mean squared error                  0.1215
    ## Relative absolute error                  2.958  %
    ## Root relative squared error             24.323  %
    ## Total Number of Instances             8124     
    ## 
    ## === Confusion Matrix ===
    ## 
    ##     a    b   <-- classified as
    ##  4208    0 |    a = e
    ##   120 3796 |    b = p

Step 4. Improving model performance
-----------------------------------

-   Boosting the accuracy of decision trees

``` r
mushroom_JRip <- JRip(type ~ ., data = mushrooms)
mushroom_JRip
```

    ## JRIP rules:
    ## ===========
    ## 
    ## (odor = f) => type=p (2160.0/0.0)
    ## (gill_size = n) and (gill_color = b) => type=p (1152.0/0.0)
    ## (gill_size = n) and (odor = p) => type=p (256.0/0.0)
    ## (odor = c) => type=p (192.0/0.0)
    ## (spore_print_color = r) => type=p (72.0/0.0)
    ## (stalk_surface_below_ring = y) and (stalk_surface_above_ring = k) => type=p (68.0/0.0)
    ## (habitat = l) and (cap_color = w) => type=p (8.0/0.0)
    ## (stalk_color_above_ring = y) => type=p (8.0/0.0)
    ##  => type=e (4208.0/0.0)
    ## 
    ## Number of Rules : 9
