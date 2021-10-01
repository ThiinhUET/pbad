from utils.casas_dataset import Casas

casas = Casas("hh101")
data = casas.get_ann_raw_dataframe()
print(data.head(50))