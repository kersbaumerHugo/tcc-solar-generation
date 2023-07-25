# tcc-solar-generation
Project used to develop my graduation final work for the degree of mechanichal engineer

## How to run my code:

First you should run `gerador_csv_geracao.py`. This code compiles every solar generation file to csv to ease the processing of the data.
Then run `gerador_dados_climaticos.py` to do the same on the climate data.

After the csv data has been assembled you can run the `merge_geracao_clima.py`, that should merge everything and create the `_full` datasets.

Next you are able to run `modelo_final.py` that will create the models and train them. Inside that file you can select to plot either the `train_test_split` plot (`plot1`), or the prediction vs real data graphs (`plot2`). You can only generate one at once. So set one variable to `True` and the other one to `False`.

Then you can run `mapa.py` that will generate the maps from the work. 

