Diagnosing breast cancer with the k-NN algorithm
================

Information
===========

Diagnosing breast cancer with the **k-NN** algorithm, data downloaded from <http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>.

Step 1. Download and prepare data
---------------------------------

-   直接从网站获取数据

``` r
wbcd <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", 
    sep = ",", header = F, stringsAsFactors = FALSE)
colnames(wbcd) <- c("id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", 
    "area_mean")
head(wbcd)
```

    ##         id diagnosis radius_mean texture_mean perimeter_mean area_mean
    ## 1   842302         M       17.99        10.38         122.80    1001.0
    ## 2   842517         M       20.57        17.77         132.90    1326.0
    ## 3 84300903         M       19.69        21.25         130.00    1203.0
    ## 4 84348301         M       11.42        20.38          77.58     386.1
    ## 5 84358402         M       20.29        14.34         135.10    1297.0
    ## 6   843786         M       12.45        15.70          82.57     477.1
    ##        NA      NA     NA      NA     NA      NA     NA     NA    NA     NA
    ## 1 0.11840 0.27760 0.3001 0.14710 0.2419 0.07871 1.0950 0.9053 8.589 153.40
    ## 2 0.08474 0.07864 0.0869 0.07017 0.1812 0.05667 0.5435 0.7339 3.398  74.08
    ## 3 0.10960 0.15990 0.1974 0.12790 0.2069 0.05999 0.7456 0.7869 4.585  94.03
    ## 4 0.14250 0.28390 0.2414 0.10520 0.2597 0.09744 0.4956 1.1560 3.445  27.23
    ## 5 0.10030 0.13280 0.1980 0.10430 0.1809 0.05883 0.7572 0.7813 5.438  94.44
    ## 6 0.12780 0.17000 0.1578 0.08089 0.2087 0.07613 0.3345 0.8902 2.217  27.19
    ##         NA      NA      NA      NA      NA       NA    NA    NA     NA
    ## 1 0.006399 0.04904 0.05373 0.01587 0.03003 0.006193 25.38 17.33 184.60
    ## 2 0.005225 0.01308 0.01860 0.01340 0.01389 0.003532 24.99 23.41 158.80
    ## 3 0.006150 0.04006 0.03832 0.02058 0.02250 0.004571 23.57 25.53 152.50
    ## 4 0.009110 0.07458 0.05661 0.01867 0.05963 0.009208 14.91 26.50  98.87
    ## 5 0.011490 0.02461 0.05688 0.01885 0.01756 0.005115 22.54 16.67 152.20
    ## 6 0.007510 0.03345 0.03672 0.01137 0.02165 0.005082 15.47 23.75 103.40
    ##       NA     NA     NA     NA     NA     NA      NA
    ## 1 2019.0 0.1622 0.6656 0.7119 0.2654 0.4601 0.11890
    ## 2 1956.0 0.1238 0.1866 0.2416 0.1860 0.2750 0.08902
    ## 3 1709.0 0.1444 0.4245 0.4504 0.2430 0.3613 0.08758
    ## 4  567.7 0.2098 0.8663 0.6869 0.2575 0.6638 0.17300
    ## 5 1575.0 0.1374 0.2050 0.4000 0.1625 0.2364 0.07678
    ## 6  741.6 0.1791 0.5249 0.5355 0.1741 0.3985 0.12440

Step 2. Transformation – normalizing numeric data
-------------------------------------------------

-   癌症数据均一化处理

``` r
wbcd <- wbcd[-1]
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"), labels = c("Benign", 
    "Malignant"))
# round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)
# summary(wbcd[c('radius_mean', 'area_mean')])

normalize <- function(x) {
    return((x - min(x))/(max(x) - min(x)))
}

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$area_mean)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.0000  0.1174  0.1729  0.2169  0.2711  1.0000

step 3. Training a model on the data
------------------------------------

-   训练模型

``` r
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

library(class)
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 21)
```

step 4. Evaluating model performance
------------------------------------

-   评估数据模型

``` r
library(gmodels)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        77 |         0 |        77 | 
    ##                  |     1.000 |     0.000 |     0.770 | 
    ##                  |     0.975 |     0.000 |           | 
    ##                  |     0.770 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        21 |        23 | 
    ##                  |     0.087 |     0.913 |     0.230 | 
    ##                  |     0.025 |     1.000 |           | 
    ##                  |     0.020 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        79 |        21 |       100 | 
    ##                  |     0.790 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

错误率仅为2%，即该模型可准确鉴定98%的肿瘤；

step 5. Improving model performance
-----------------------------------

-   提升模型鉴别能力

``` r
# z-score standardization
wbcd_z <- as.data.frame(scale(wbcd[-1]))
summary(wbcd_z$area_mean)  # To confirm that the transformation was applied correctly;
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -1.4532 -0.6666 -0.2949  0.0000  0.3632  5.2459

``` r
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ]
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 21)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        77 |         0 |        77 | 
    ##                  |     1.000 |     0.000 |     0.770 | 
    ##                  |     0.975 |     0.000 |           | 
    ##                  |     0.770 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        21 |        23 | 
    ##                  |     0.087 |     0.913 |     0.230 | 
    ##                  |     0.025 |     1.000 |           | 
    ##                  |     0.020 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        79 |        21 |       100 | 
    ##                  |     0.790 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

z-score标准化和normalize均一化数据无差异；

step 6. Testing alternative values of k
---------------------------------------

-   尝试多个k值评估错误率

### k = 1

``` r
wbcd_test_pred_1 <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 1)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_1, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred_1 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        73 |         4 |        77 | 
    ##                  |     0.948 |     0.052 |     0.770 | 
    ##                  |     0.973 |     0.160 |           | 
    ##                  |     0.730 |     0.040 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        21 |        23 | 
    ##                  |     0.087 |     0.913 |     0.230 | 
    ##                  |     0.027 |     0.840 |           | 
    ##                  |     0.020 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        75 |        25 |       100 | 
    ##                  |     0.750 |     0.250 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

### k = 5

``` r
wbcd_test_pred_5 <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 5)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_5, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred_5 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        73 |         4 |        77 | 
    ##                  |     0.948 |     0.052 |     0.770 | 
    ##                  |     1.000 |     0.148 |           | 
    ##                  |     0.730 |     0.040 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         0 |        23 |        23 | 
    ##                  |     0.000 |     1.000 |     0.230 | 
    ##                  |     0.000 |     0.852 |           | 
    ##                  |     0.000 |     0.230 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        73 |        27 |       100 | 
    ##                  |     0.730 |     0.270 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

### k = 11

``` r
wbcd_test_pred_11 <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 11)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_11, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred_11 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        76 |         1 |        77 | 
    ##                  |     0.987 |     0.013 |     0.770 | 
    ##                  |     0.987 |     0.043 |           | 
    ##                  |     0.760 |     0.010 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         1 |        22 |        23 | 
    ##                  |     0.043 |     0.957 |     0.230 | 
    ##                  |     0.013 |     0.957 |           | 
    ##                  |     0.010 |     0.220 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        77 |        23 |       100 | 
    ##                  |     0.770 |     0.230 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

### k = 15

``` r
wbcd_test_pred_15 <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 15)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_15, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred_15 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        77 |         0 |        77 | 
    ##                  |     1.000 |     0.000 |     0.770 | 
    ##                  |     0.975 |     0.000 |           | 
    ##                  |     0.770 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        21 |        23 | 
    ##                  |     0.087 |     0.913 |     0.230 | 
    ##                  |     0.025 |     1.000 |           | 
    ##                  |     0.020 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        79 |        21 |       100 | 
    ##                  |     0.790 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ## 

### k = 27

``` r
wbcd_test_pred_27 <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, 
    k = 27)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred_27, prop.chisq = FALSE)
```

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  100 
    ## 
    ##  
    ##                  | wbcd_test_pred_27 
    ## wbcd_test_labels |    Benign | Malignant | Row Total | 
    ## -----------------|-----------|-----------|-----------|
    ##           Benign |        77 |         0 |        77 | 
    ##                  |     1.000 |     0.000 |     0.770 | 
    ##                  |     0.975 |     0.000 |           | 
    ##                  |     0.770 |     0.000 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##        Malignant |         2 |        21 |        23 | 
    ##                  |     0.087 |     0.913 |     0.230 | 
    ##                  |     0.025 |     1.000 |           | 
    ##                  |     0.020 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ##     Column Total |        79 |        21 |       100 | 
    ##                  |     0.790 |     0.210 |           | 
    ## -----------------|-----------|-----------|-----------|
    ## 
    ##
