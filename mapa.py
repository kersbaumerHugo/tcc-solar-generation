import pandas as pd
import openpyxl
from difflib import get_close_matches
import os
import csv
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
def gerar_plot_usinas():
  file_info_geral = cwd+r'\\info_geral.csv'
  with open(file_info_geral, 'r',encoding="utf-8") as f:
        # Create a CSV reader object
        csv_reader = csv.reader(f,delimiter=',')
        csv_reader = list(csv_reader)
        df_info_geral = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
  print(df_info_geral.columns)
  print(df_info_geral)
  #df_info_geral = df_info_geral.drop(columns=["'"])
  df_info_geral = df_info_geral.sort_values(["Total de dias","Dias com Geração == 0","Dias radiação == 0"],ascending=[False,True,False])
  
  lats = df_info_geral['Latitude'].apply(lambda x: float(x)).to_list()
  lons = df_info_geral['Longitude'].apply(lambda x: float(x)).to_list()
  # draw map with markers for float locations
  m = Basemap(projection='merc')
  x, y = m(lons,lats)
  m.drawmapboundary(fill_color='#99ffff')
  m.drawcountries()
  m.drawstates(color="gray")
  m.fillcontinents(color='#cc9966',lake_color='#99ffff')
  m.scatter(x,y,3,marker='o',color='k')
  plt.title('Distribuição das %d plantas pelo Brasil' %(len(lats)),fontsize=12)
  print(df_info_geral)
  return m
def gerar_plot_estacoes():
  file_info_geral = cwd+r'\\tables\\estacoes_lat_long.csv'
  with open(file_info_geral, 'r',encoding="utf-8") as f:
        # Create a CSV reader object
        csv_reader = csv.reader(f,delimiter=';')
        csv_reader = list(csv_reader)
        df_info_geral = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
  lats = df_info_geral['Lat'].apply(lambda x: float(x)).to_list()
  lons = df_info_geral['long'].apply(lambda x: float(x)).to_list()
  m = Basemap(projection='merc')
  x, y = m(lons,lats)
  m.drawmapboundary(fill_color='#99ffff')
  m.drawcountries()
  m.drawstates(color="gray")
  m.fillcontinents(color='#cc9966',lake_color='#99ffff')
  m.scatter(x,y,3,marker='x',color='blue')
  plt.title('Distribuição das %d estações pelo Brasil' %(len(lats)),fontsize=12)
gerar_dados = False
cwd = os.getcwd()
if gerar_dados:

  file_list = os.listdir(cwd+r'\\geracao_csv')
  all_geracao = []
  coords = {}
  header = ["Usina","Total de dias","Dias com Geração == 0","Dias com Geração NaN","Start Date","End Date","Dias radiação == 0","Dias radiação NaN","Estação escolhida"]
  data=[]
  for file in file_list:
      if "geracao.csv" in file:
        estacao_name = file[:-12]
        dataset_full = open_geracao(estacao_name)
        dados = get_all_data_df(estacao_name,dataset_full)
        if dados != []:
          data.append(dados)
          dataset_full.to_csv(cwd+r'\\full\\'+estacao_name+"_full.csv")
  df_info_geral = pd.DataFrame(data,columns=header)
  df_info_geral.sort_values(["Total de dias","Dias com Geração == 0"],ascending=[False,True])
  print(df_info_geral)
else:
  m = gerar_plot_usinas()
  n = gerar_plot_estacoes()
  plt.show()
