import pandas as pd
import openpyxl
from difflib import get_close_matches
import os
import csv
import time
import matplotlib.pyplot as plt
def rmse(y_true,y_pred):
  return mse(y_true,y_pred,squared=False)
def replace_decimal(val):
    return str(val).replace(',', '.')
def open_geracao(file_name):
    file_geracao = file_name+"_geracao.csv"
    file_cilma = file_name+"_clima.csv"
    with open(file_geracao, 'r',encoding="utf-8") as f:
        # Create a CSV reader object
        csv_reader = csv.reader(f)
        csv_reader = list(csv_reader)
        df_geracao = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
    df_geracao["Hora"] = df_geracao["Hora"].apply(lambda x: str(float(x) % 24)[:-2]+":00:00")
    df_geracao["Dia"] = df_geracao["Dia"].apply(lambda x: str(x)[:-9] if len(str(x))>10 else "2021"+str(x)[:-9])
    df_geracao["Date"] = df_geracao["Dia"]+" "+df_geracao["Hora"]
    df_geracao.drop(columns=["Dia","Hora"])
    df_geracao["Date"] = pd.to_datetime(df_geracao["Date"])
    df_geracao.set_index("Date",inplace=True)
    df_geracao = df_geracao.apply(lambda x: x.str.replace(',', '.'))
    with open(file_cilma, 'r',encoding="utf-8") as g:
        csv_reader = csv.reader(g)
        csv_reader = list(csv_reader)
        df_clima = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
    contador=0
    df_clima["Hora UTC"+str(contador)] = df_clima["Hora UTC"+str(contador)].apply(lambda x: x[:2]+":"+x[2:-4]+":"+"00")
    df_clima["Date"+str(contador)] = df_clima["Data"+str(contador)]+" "+df_clima["Hora UTC"+str(contador)]
    #df_clima = df_clima.drop(columns=["Data"+str(contador),"Hora UTC"+str(contador)])
    df_clima["Date"+str(contador)] = pd.to_datetime(df_clima["Date"+str(contador)])
    df_clima.set_index("Date"+str(contador),inplace=True)
    #print(df_clima.columns)
    for contador in range(0,3):
      df_clima["RADIACAO GLOBAL (Kj/m²)"+str(contador)] = df_clima["RADIACAO GLOBAL (Kj/m²)"+str(contador)].apply(lambda x: 0.0 if (x.replace(",",".")=="" or float(x.replace(",","."))<0) else float(x.replace(",",".")))
      df_clima = df_clima.rename(columns={'RADIACAO GLOBAL (Kj/m²)'+str(contador): 'radiation'+str(contador)})
      df_clima = df_clima.drop(columns=["Data"+str(contador),"Hora UTC"+str(contador)])
      df_clima = df_clima.applymap(replace_decimal)
      df_clima['radiation'+str(contador)] = df_clima['radiation'+str(contador)].apply(lambda x: 0 if x=="" else float(x))
      df_clima = df_clima[df_clima['radiation'+str(contador)]>0]
    
    dataset_full = df_clima.merge(df_geracao,left_index=True,right_index=True)
    
    dataset_full = dataset_full.drop(columns=['nomes_match', 'nome_geracao', 'nome_ceg', 'nome_upper',
       'DatGeracaoConjuntoDados', 'NomEmpreendimento', 'IdeNucleoCEG',
       'CodCEG', 'SigUFPrincipal', 'SigTipoGeracao', 'DscFaseUsina',
       'DscOrigemCombustivel', 'DscFonteCombustivel', 'DscTipoOutorga',
       'NomFonteCombustivel', 'DatEntradaOperacao', 'MdaPotenciaOutorgadaKw',
       'MdaPotenciaFiscalizadaKw', 'MdaGarantiaFisicaKw', 'DatInicioVigencia', 'DatFimVigencia',
       'DscPropriRegimePariticipacao', 'DscSubBacia', 'DscMuninicpios', '_x', '_y'])
    dataset_full = dataset_full.fillna(0)
    print(dataset_full.columns)
    print(dataset_full)
    dates = dataset_full.index
    value1 = dataset_full["Geração no Centro de Gravidade - MW médios (Gp,j) - MWh"]
    value2 = dataset_full["radiation0"]
    plt.plot(value1.index, value1, label='geracao')
    #plt.plot(value2.index, value2, label='radiacao')
    plt.title('Comparison of Two Values over Time')
    plt.xlabel('Date')
    plt.ylabel('Radiation/Generation')
    # Add legend
    plt.legend()

    # Show plot
    plt.show()
    return(dataset_full)
dataset_full=open_geracao("PIRAPORA 5")
