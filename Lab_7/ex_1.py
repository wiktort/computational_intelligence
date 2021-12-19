import pandas as pd
from matplotlib import pyplot as plt
import re
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# get dataset
dataset_url = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
df = pd.read_csv(dataset_url)

# clean up
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

french_basket = (df[df['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))


# draw barChart with
bar_chart_array = []
for column_name in french_basket.columns:
    bar_chart_array.append([column_name, french_basket[column_name].sum()])
bar_chart = (pd.DataFrame(data=bar_chart_array, columns=["Product name", "Total quantity"])
             .sort_values(by=["Total quantity"], ascending=False))

y = []
x = []
for i in range(10):
    y.append(bar_chart.iloc[i]["Total quantity"])
    x.append(bar_chart.iloc[i]["Product name"])
plt.figure(1)
plt.bar(x, y)
plt.xticks(rotation='vertical')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)


# transform
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


basket_sets = french_basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

# get rules
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("Rules with min_support == 5%:\n{}\n".format(rules))
rules_with_min_confidence = rules['confidence'] >= 0.60
print("Rules with min_confidence == 60%:\n{}\n".format(rules[rules_with_min_confidence]))

# sort rules by confidence
rules_sorted = rules.sort_values(by=["confidence"], ascending=False)


# create function that returns rules with provided product
def get_rules_with_product(product_name: str):
    return rules[(rules['antecedents'] == {product_name}) | (rules['consequents'] == {product_name})]


print("Get rules that contains 'ALARM CLOCK BAKELIKE PINK' in either 'antecedents' or 'consequents': )\n")
print(get_rules_with_product("ALARM CLOCK BAKELIKE PINK"))


# Compare confidence and support
x1 = rules['confidence']
y1 = rules['support']

plt.figure(2)
plt.plot(x1, y1, 'go')
plt.ylabel(ylabel='confidence')
plt.xlabel(xlabel='support')
plt.show()