import pandas as pd
import openpyxl
from difflib import get_close_matches
import os
import csv
import time
import datetime as dt
import numpy as np
def replace_decimal(val):
    return str(val).replace(',', '.')
def day_to_date(day_number, year):
    date_format = '%Y-%m-%d'
    # create a datetime object for January 1st of the given year
    start_date = dt.datetime(year, 1, 1)
    # add the number of days to the start date
    result_date = start_date + dt.timedelta(days=day_number-2)
    # format the date string using the specified format
    #print(result_date)
    return result_date
def open_geracao(file_name):
    cwd = os.getcwd()
    geracao_folder = cwd+r'\\geracao_csv\\'
    clima_folder = cwd+r'\\clima_csv\\'
    file_geracao = geracao_folder+file_name+"_geracao.csv"
    file_cilma = clima_folder+file_name+"_clima.csv"
    with open(file_geracao, 'r',encoding="utf-8") as f:
        # Create a CSV reader object
        csv_reader = csv.reader(f)
        csv_reader = list(csv_reader)
        df_geracao = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
        print(file_name)
    df_geracao["Hora"] = df_geracao["Hora"].apply(lambda x: " "+str(float(x) % 24)[:-2]+":00:00")
    print(df_geracao["Dia"])
    #df_geracao["Dia"] = pd.to_datetime(df_geracao["Dia"])
    df_geracao["Dia"] = df_geracao["Dia"].apply(lambda x: str(day_to_date(int(x),1900)) if len(x)<8 else x)
    df_geracao["Dia"] = df_geracao["Dia"].apply(lambda x: str(x)[:-9] )#if len(str(x))>10 else "2021"+str(x)[:-9])
    #df_geracao["Dia"] = pd.to_datetime(df_geracao["Dia"])
    try:
        df_geracao["Date"] = pd.to_datetime(df_geracao["Dia"]+df_geracao["Hora"],format="%Y-%m-%d %H:%M:%S")
    except ValueError:
        if file_name =="ALEX III":
            print(df_geracao[df_geracao["Dia"]=="44711"]["Dia"])
            for dia in df_geracao["Dia"].unique():
                #if dia > "2022-05-30" and dia < "2022-06-01":
                    print(dia)
            asdasd
            print(df_geracao[df_geracao["Dia"]=="31/05/2022"])
            df_geracao["Date"] = pd.to_datetime(str(pd.to_datetime(df_geracao["Dia"],origin=""))+df_geracao["Hora"],format="%Y-%m-%d %H:%M:%S")
    df_geracao.drop(columns=["Dia","Hora"])
    df_geracao["Date"] = pd.to_datetime(df_geracao["Date"])
    df_geracao.set_index("Date",inplace=True)
    df_geracao = df_geracao.apply(lambda x: x.str.replace(',', '.'))
    with open(file_cilma, 'r',encoding="utf-8") as g:
        csv_reader = csv.reader(g)
        csv_reader = list(csv_reader)
        df_clima = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
    contador=0
    #print(file_name,df_clima.columns)
    #df_clima["Hora UTC"+str(contador)] = df_clima["Hora UTC"+str(contador)].apply(lambda x: x[:2]+":"+x[2:-4]+":"+"00")
    #df_clima["Date"+str(contador)] = df_clima["Data"+str(contador)]+" "+df_clima["Hora UTC"+str(contador)]
    #df_clima = df_clima.drop(columns=["Data"+str(contador),"Hora UTC"+str(contador)])
    df_clima["Date"+str(contador)] = pd.to_datetime(df_clima["Date"+str(contador)])
    df_clima.set_index("Date"+str(contador),inplace=True)
    #print(df_clima.columns)
    for contador in range(0,3):
      #df_clima["RADIACAO GLOBAL (Kj/m²)"+str(contador)] = df_clima["RADIACAO GLOBAL (Kj/m²)"+str(contador)].apply(lambda x: 0.0 if (x.replace(",",".")=="" or float(x.replace(",","."))<0) else float(x.replace(",",".")))
      df_clima = df_clima.rename(columns={'RADIACAO GLOBAL (Kj/m²)'+str(contador): 'radiation'+str(contador)})
      df_clima = df_clima.drop(columns=[str(contador)])
      df_clima = df_clima.applymap(replace_decimal)
      #df_clima['radiation'+str(contador)] = df_clima['radiation'+str(contador)].apply(lambda x: 0 if x=="" else float(x))#Não fazer isso pois remove a quantidade de N/A
      #df_clima = df_clima[df_clima['radiation'+str(contador)]>0]#Tirar apenas se os 3 forem 0
    dataset_full = df_clima.merge(df_geracao,left_index=True,right_index=True)
    #Drop de colunas desnecessárias do DS de geração
    dataset_full = dataset_full.drop(columns=['nomes_match', 'nome_geracao', 'nome_ceg', 'nome_upper',
       'DatGeracaoConjuntoDados', 'NomEmpreendimento', 'IdeNucleoCEG',
       'CodCEG', 'SigUFPrincipal', 'SigTipoGeracao', 'DscFaseUsina',
       'DscOrigemCombustivel', 'DscFonteCombustivel', 'DscTipoOutorga',
       'NomFonteCombustivel', 'DatEntradaOperacao', 'MdaPotenciaOutorgadaKw',
       'MdaPotenciaFiscalizadaKw', 'MdaGarantiaFisicaKw', 'DatInicioVigencia', 'DatFimVigencia',
       'DscPropriRegimePariticipacao', 'DscSubBacia', 'DscMuninicpios'])#, '_x', '_y','0','1','2'])
    
    #dataset_full = dataset_full.fillna(0)
    #print(dataset_full.columns)
    #print(dataset_full)
    return(dataset_full)
def escolhe_estacao(df,num_dias):
  dias_zero = {}
  dias_NaN = {}
  for estacao in range(0,3):
    falha = False
    estacao = str(estacao)
    dias_zero[estacao] = 0
    dias_NaN[estacao] = 0
    count = 0
    #print(df.columns)
    for column in df.columns:
      #print(estacao,column[-1:],column[-1:]==estacao)
      if column[-1:] == estacao:
        #print(column[:-1] != "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)")
        if column[:-1] != "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)":
          dias_zero[column] = df.isin([0]).sum(axis=0)[column]
          dias_NaN[column] = df.isin([np.NaN]).sum(axis=0)[column]
        else:
          dias_zero[column] = 0
          dias_NaN[column] = 0
        falha_zero = (dias_zero[column]/num_dias)>0.75
        falha_NaN = (dias_NaN[column]/num_dias)>0.75
        falha = falha_zero or falha_NaN or falha
        #print(not falha)
    #print(not falha)
    if not falha :
      return estacao
  return -1     
def get_all_data_df(estacao_name,df):
  #print(df.columns)
  for column in df.columns:
    if column not in ["Sigla da Usina","Hora","Dia","IdcGeracaoQualificada"]:
      #print(column)
      df[column] = df[column].apply(lambda x: float(x) if (x != "") else np.NaN)
  df_por_dia = df.resample("D").sum()
  #print(df_por_dia)
  num_dias = len(df_por_dia.index)
  choice_column_names = []
  for column in df.columns:
    if column[-1:] == "0":
      choice_column_names.append(column[:-1])
  escolhida = escolhe_estacao(df_por_dia,num_dias)
  print(escolhida)
  removed = [0,1,2]
  for estacao in removed:
    for column in choice_column_names:
      if str(estacao) == escolhida:
        df_por_dia.rename(columns={column+escolhida:column},inplace=True)
        df.rename(columns={column+escolhida:column},inplace=True)
      else:
        df_por_dia.drop(columns=[column+str(estacao)],inplace=True)
        df.drop(columns=[column+str(estacao)],inplace=True)
  ##print(df_por_dia.columns)
  #print(escolhida)
  lat = df["NumCoordNEmpreendimento"][0]
  long = df["NumCoordEEmpreendimento"][0]
  print(lat)
  print(long)
  dias_zero_geracao = df_por_dia.isin([0]).sum(axis=0)["Geração no Centro de Gravidade - MW médios (Gp,j) - MWh"]
  dias_NaN_geracao = df_por_dia.isin([np.NaN]).sum(axis=0)["Geração no Centro de Gravidade - MW médios (Gp,j) - MWh"]
  dias_zero_radiation = df_por_dia.isin([0]).sum(axis=0)["radiation"]
  dias_NaN_radiation = df_por_dia.isin([np.NaN]).sum(axis=0)["radiation"]
  start_date = df_por_dia.index.min()
  end_date = df_por_dia.index.max()
  to_remove = ["UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)","UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)","VENTO, DIREÇÃO HORARIA (gr) (° (gr))","VENTO, RAJADA MAXIMA (m/s)",
             "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)","PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)","TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
             "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)","TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)","TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
             "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)","UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)","Sigla da Usina","Hora","Dia","IdcGeracaoQualificada"]
  df = df.drop(to_remove,axis=1)
  
  data = [estacao_name,num_dias,dias_zero_geracao,dias_NaN_geracao,start_date,end_date,dias_zero_radiation,dias_NaN_radiation,escolhida,lat,long]
  
  
  return data,df
gerar_dados = True
cwd = os.getcwd()
if gerar_dados:
  file_list = os.listdir(cwd+r'\\geracao_csv')
  all_geracao = []
  coords = {}
  header = ["Usina","Total de dias","Dias com Geração == 0","Dias com Geração NaN","Start Date","End Date","Dias radiação == 0","Dias radiação NaN","Estação escolhida","Latitude","Longitude"]
  data=[]
  for file in file_list:
      if "geracao.csv" in file:
        estacao_name = file[:-12]
        dataset_full = open_geracao(estacao_name)
        #print(dataset_full)
        dados,dataset_full = get_all_data_df(estacao_name,dataset_full)
        if dados == []:
          print("Dataset "+estacao_name+" ignorado devido a excesso de dados faltantes")
        else:
          data.append(dados)
          dataset_full.to_csv(cwd+r'\\full\\'+estacao_name+"_full.csv")
  df_info_geral = pd.DataFrame(data,columns=header)
  df_info_geral.sort_values(["Total de dias","Dias com Geração == 0","Dias radiação == 0"],ascending=[False,True,False])
  df_info_geral.to_csv(cwd+r'\\info_geral.csv')
  print(df_info_geral)
else:
  file_info_geral = cwd+r'\\info_geral.csv'
  with open(file_info_geral, 'r') as f:
        # Create a CSV reader object
        csv_reader = csv.reader(f,delimiter=';')
        csv_reader = list(csv_reader)
        df_info_geral = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
  df_info_geral = df_info_geral.drop(columns=["Column1"])
  df_info_geral = df_info_geral.sort_values(["Total de dias","Dias com Geração == 0","Dias radiação == 0"],ascending=[False,True,False])
  print(df_info_geral)
