Diagnosing breast cancer with the k-NN algorithm
================

Information
===========

Diagnosing breast cancer with the **k-NN** algorithm, data downloaded from <http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/>.

Step 1. Download and prepare data
---------------------------------

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

step 3. Data preparation – creating training and test datasets
==============================================================
