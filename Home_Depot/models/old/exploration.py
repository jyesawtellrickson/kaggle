# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import io

from subprocess import check_output
print(check_output(["ls","../input/"]).decode("utf8"))

# load files
training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")

# convert product titel to utf-8
#training_data["product_title"] = iconv(training_data$product_title, "UTF-8", "ASCII", "")


# merge descriptions - bring descriptions into training data
training_data = pd.merge(training_data, descriptions, on="product_uid", how="left")

# merge product counts
product_counts = pd.DataFrame(pd.Series(training_data.groupby(["product_uid"]).size(), name="product_count"))
training_data = pd.merge(training_data, product_counts, left_on="product_uid", right_index=True, how="left")

# merge brand names
# find brand names of the items and then add to the training data
# all unknown brand names should be "Unknown"
brand_names = attribute_data[attribute_data.name == "MFG Brand Name"] \
    [["product_uid", "value"]].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")
training_data.brand_name.fillna("Unknown", inplace=True)

# training set processed, let's have a look
print(str(training_data.info()))
print(str(training_data.describe()))
training_data[:50]

# look at types of attributes and indoor/outdoor propensity
print(attribute_data.name.value_counts())
print(attribute_data.value[attribute_data.name == "Indoor/Outdoor"].value_counts())

# group the ids into bins(?) and perform correlation
# check if any of the columns are correlated
training_data["id_bins"] = pd.cut(training_data.id, 20, labels=False)
print(training_data.corr(method="spearman"))
training_data.describe()

# check the relevance data we have
training_data.relevance.hist()
training_data.relevance.value_counts()


# plot scores
un_rel, co_rel = np.unique(training_data.relevance, return_counts=True)



# map the ones that are wrong so that we can visualise


# post process to maxe the distributions match


# We seem to have a lot more very relevant searches.
# What about typos?