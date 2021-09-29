from src.utils.casas_dataset import Casas

casas = Casas()
data = Casas.get_ann_features("hh101")
print(data)