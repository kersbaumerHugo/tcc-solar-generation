import pandas as pd
import openpyxl
from difflib import get_close_matches
import os
import csv
import time
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

def rmse(y_true,y_pred):
  return mse(y_true,y_pred,squared=False)
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
def detect_date(date,amount):
    #print(date,date>pd.to_datetime("24/02/2023 23:00:00"))
    if date-dt.timedelta(amount)>pd.to_datetime("01/09/2020 23:00:00"):
        return date+dt.timedelta(amount)
    else:
        return date-dt.timedelta(amount)
def na_treatment (df,column,treatment,total,file):
    na_lines = df[df[column].isin([np.NaN])]
    #Removendo horas maiores que 20h da noite
    df = df[df.index.hour<=pd.to_datetime("01/01/1900 22:00:00").hour]
    na_lines = na_lines[na_lines.index.hour<=pd.to_datetime("01/01/1900 21:00:00").hour]
    #Removendo horas menores que 05h da manhã
    df = df[df.index.hour>=pd.to_datetime("01/01/1900 09:00:00").hour]
    na_lines = na_lines[na_lines.index.hour>=pd.to_datetime("01/01/1900 08:00:00").hour]#.resample("D").sum()
    for column in df.columns:
        df[column] = df[column].apply(lambda x: float(x) if (x != "") else np.NaN)
    for column in na_lines.columns:
        na_lines[column] = na_lines[column].apply(lambda x: float(x) if (x != "") else np.NaN)
    na_lines = na_lines.resample("D").sum()#Mudar para fazer a análise por hora ao invés de por dia
    df_sum = df.resample("D").sum()
    #print(len(na_lines),len(df)*0.8,len(na_lines) <= len(df)*0.8)
    if len(na_lines) <= len(df_sum)*0.3:
        #print(na_lines)
        #print(df)
        file_list.append(file)
    else:
        print("Impossível remover NA's pois não há dados para horários importantes do dia")
        total-=1
        #na_lines = na_lines.resample("D").sum()
        
        #print(na_lines)
        #print(na_lines.loc[df.index[0]])
        #for days in range(1,7):
        #print(df.loc[na_lines.index.to_series().apply(lambda x: detect_date(x,7))]["RADIACAO"])
        #asdasd
    return total,file_list,df
t0_exec = time.time()
cwd = os.getcwd()
last_sum=[[]]
get_NaN = True
total = 111
file_list = []
plot1=False
plot2=True
if get_NaN:
    for file in os.listdir(cwd+r'\\full\\'):
        file_info_geral = cwd+r'\\full\\'+file#AcIII #Coremas II e III #Lavras 
        with open(file_info_geral, 'r',encoding="utf-8") as f:
        # Create a CSV reader object
            csv_reader = csv.reader(f,delimiter=',')
            csv_reader = list(csv_reader)
            df_full = pd.DataFrame(columns=csv_reader[0],data=csv_reader[1:])
        index = df_full.iloc[:, 0]
        #print(df_full.columns)
        df_full = df_full.set_index(pd.to_datetime(index))
        df_full.drop(labels=df_full.columns[0],inplace=True,axis=1)
        df_full = df_full.replace(r'^\s*$', np.nan, regex=True)
        df_full = df_full.rename(columns={"Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": 'GERACAO'})#PV production
        df_full = df_full.rename(columns={"PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": 'PRECIPITACAO'})#tirar
        df_full = df_full.rename(columns={"PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": 'PATM'})#tirar
        df_full = df_full.rename(columns={"TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": 'TEMPERATURA'})#temperature
        df_full = df_full.rename(columns={"TEMPERATURA DO PONTO DE ORVALHO (°C)": 'TEMPERATURA_ORVALHO'})#tirar
        df_full = df_full.rename(columns={"UMIDADE RELATIVA DO AR, HORARIA (%)": 'UMIDADE'})#humidity
        df_full = df_full.rename(columns={"VENTO, VELOCIDADE HORARIA (m/s)": 'VEL_VENTO'})#tirar
        df_full = df_full.rename(columns={"NumCoordNEmpreendimento": 'COORD_N'})#tirar (interessante para caso entre usinas)
        df_full = df_full.rename(columns={"NumCoordEEmpreendimento": 'COORD_E'})#tirar (interessante para caso entre usinas)
        df_full = df_full.rename(columns={"radiation": 'RADIACAO'})#solar irradiance
        #temperature, humidity, solar irradiance, PV production,
        nan_percent = df_full.isin([np.NaN]).sum(axis=0).apply(lambda x: (x/len(df_full))*100)
        if nan_percent.sum() >0:
          #print(nan_percent)
          print(file[:-9])
          total,file_list,df_full = na_treatment(df_full,"RADIACAO","7days",total,file)
          #total,file_list,df_full = na_treatment(df_full,"PRECIPITACAO","7days",total,file)
        last_sum[0] = ["File","Len",*df_full.columns,"df_full"]
        if len(df_full)>=000 :#and (df_full.isin([np.NaN]).sum(axis=0).sum()<(40*24)): 
            last_sum.append([file,len(df_full),*nan_percent,df_full])
    df_NaN = pd.DataFrame(columns=last_sum[0],data=last_sum[1:])
    #print(total)
    #print(file_list)
    #print(last_sum)
    #print(df_NaN.sort_values(["RADIACAO"],ascending=[True]).reset_index())
df_all_trees = []
df_all_forests = []
min_start_date = dt.datetime.today()
plot_usinas_group = {}
max_xtick_labels = []
ranges = []
group_counter = 0
previsoes = []
for usina in range(1,len(last_sum[:])):#Trocar para file_list
  df_full = last_sum[usina][-1]
  nome_usina = last_sum[usina][0][:-9]
  
  print(nome_usina+" "+str(usina/len(last_sum)*100)+"%")
  print(df_full)
  #df_full = df_full.replace(r'^\s*$', np.nan, regex=True)
  df_full["Mes"] = df_full.index.month
  print(df_full["Mes"])
  #print(df_full[df_full["Mes"].isin([np.NaN])])
  months_encoder = LabelBinarizer()
  months_encoder.fit(df_full['Mes'])
  transformed = months_encoder.transform(df_full['Mes'])
  ohe_df = pd.DataFrame(transformed,index=pd.to_datetime(df_full.index))
  df_full = df_full.merge(ohe_df,left_index=True,right_index=True)#onehot mes
  df_full = df_full.drop(['Mes'], axis=1)
  df_full["Hour"] = df_full.index.hour.astype(int)
  df_full.columns = df_full.columns.astype(str)
  #df_full = df_full.replace(r'^\s*$', np.nan, regex=True)
  #print(df_full.isin([np.NaN]).sum(axis=0))
  #print(df_full)
  #print(df_full["Hour"])
  #df_full["hour_sin"] = sin_transformer(24).fit_transform(df_full["Hour"])
  #Cosseno faz mais sentido deixar ele inversamente proporcional a geração
  df_full["hour_cos"] = cos_transformer(24).fit_transform(df_full["Hour"])#hour cos
  df_full = df_full.drop(['Hour'], axis=1)
  #df_full["PRECIPITACAO"] = df_full["PRECIPITACAO"].fillna(0)
  #df_full["PATM"] = df_full["PATM"].fillna(df_full["PATM"].mean())
  df_full = df_full.dropna(subset=["RADIACAO"],axis=0)
  df_full = df_full.fillna(df_full["TEMPERATURA"].mean(),axis=0)
  na_lines = df_full[df_full.isin([np.NaN])]
  na_lines = na_lines.sum()
  #print(df_full)
  #print(na_lines)
  columns_to_norm = df_full.columns.to_list()
  #print(columns_to_norm)
  scaler = MinMaxScaler() 
  arr_scaled = scaler.fit_transform(df_full)
  df_full = pd.DataFrame(arr_scaled, columns=df_full.columns,index=df_full.index)
  df_full = df_full.rename(columns={"Geração no Centro de Gravidade - MW médios (Gp,j) - MWh": 'GERACAO'})
  df_full = df_full.rename(columns={"PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": 'PRECIPITACAO'})#Imprimir por dataset quantidade de NaN por feature
  df_full = df_full.rename(columns={"PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": 'PATM'})
  df_full = df_full.rename(columns={"TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": 'TEMPERATURA'})
  df_full = df_full.rename(columns={"TEMPERATURA DO PONTO DE ORVALHO (°C)": 'TEMPERATURA_ORVALHO'})
  df_full = df_full.rename(columns={"UMIDADE RELATIVA DO AR, HORARIA (%)": 'UMIDADE'})
  df_full = df_full.rename(columns={"VENTO, VELOCIDADE HORARIA (m/s)": 'VEL_VENTO'})
  df_full = df_full.rename(columns={"NumCoordNEmpreendimento": 'COORD_N'})
  df_full = df_full.rename(columns={"NumCoordEEmpreendimento": 'COORD_E'})
  df_full = df_full.rename(columns={"radiation": 'RADIACAO'})
  df_full = df_full.drop(columns=["VEL_VENTO","COORD_N","COORD_E","TEMPERATURA_ORVALHO","PATM","PRECIPITACAO"])
  total_NaN_count_radiation = df_full.isin([np.NaN]).sum()["RADIACAO"]
  total_NaN_count = df_full.isin([np.NaN]).sum().sum()
  df_full['GERACAO'] = df_full['GERACAO'].astype("float",errors="raise")#Imprimir por dataset quantidade de NaN por feature
  df_full['TEMPERATURA'] = df_full['TEMPERATURA'].astype("float",errors="raise")
  df_full['UMIDADE'] = df_full['UMIDADE'].astype("float",errors="raise")
  df_full['RADIACAO'] = df_full['RADIACAO'].astype("float",errors="raise")
  df_full.sort_index(inplace=True)
  print(df_full)
  for n in range(11,-1,-1):
      df_full = df_full.rename(columns={str(n): str(n+1)})
  print(na_lines)
  #print(df_full.columns)

  #print(df_full["GERACAO"])
  y = df_full['GERACAO']
  X = df_full.drop('GERACAO',axis=1)
  if len(X)>10000:
    print(total_NaN_count_radiation)
    print(X[X.isin([np.NaN])].sum())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train["UMIDADE"].fillna(X_train["UMIDADE"].mean(),inplace=True)
    X_test["UMIDADE"].fillna(X_train["UMIDADE"].mean(),inplace=True)
    print("Xtrain")
    print(X_train)
    print("Xtest")
    print(X_test)
    if plot1:
      indice_train = X_train.resample("D").sum().index.unique().to_list() 
      indice_test = X_test.resample("D").sum().index.unique().to_list()
      xtick_labels = X.resample("M").sum().index.unique()[1:].insert(0,indice_train[0]).to_list()#.insert(-2,indice_test[0]).to_list()
      #print(indice_train[-1]-indice_train[0],xticks)
      start_date = indice_train[0]
      end_date = indice_test[-1]
      if [start_date,end_date] not in ranges:
        ranges.append([start_date,end_date])
      group = ranges.index([start_date,end_date])
      print(group)  
      plot_usinas_group[nome_usina] = [start_date,end_date,group,indice_train,indice_test,xtick_labels]
      if len(xtick_labels) > len(max_xtick_labels):
          max_xtick_labels = xtick_labels

    
    #Parâmetros a serem variados e seus valores
    grid = {
      'max_features': [1.0],
      'max_depth' : [5,7,10,12,15,20],
      'criterion' :['squared_error'],
      'random_state' : [42]}
    tscv = TimeSeriesSplit()
    dt_cv = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=grid, cv=tscv ,scoring="r2",return_train_score=True)
    if nome_usina == "SOL DO FUTURO I":
      start_train,end_train,start_val,end_val = [],[],[],[]
      for i, (train_index, val_index) in enumerate(tscv.split(X_train)):
        start_train.append(train_index[0])
        end_train.append(train_index[-1])
        start_val.append(val_index[0])
        end_val.append(val_index[-1])
      tss = [start_train,end_train,start_val,end_val]
      print(tss)
    best_usinas = ["SOL DO FUTURO I","SOL DO FUTURO II","SOL DO FUTURO III"]    
    t_0 = time.time()
    dt_cv.fit(X_train, y_train)
    t_f = time.time() - t_0
    print("Fit_time =",t_f)
    t_0 = time.time()
    trees = pd.DataFrame(dt_cv.cv_results_)
    print(trees)
    best_score = dt_cv.best_score_
    best_model = dt_cv.best_estimator_
    best_params = dt_cv.best_estimator_
    model_predict_dt = best_model.predict(X_test)
    if nome_usina in best_usinas:
      previsoes.append(["árvore de decisão",nome_usina,y_test,pd.DataFrame(model_predict_dt,index=y_test.index,columns=["GERACAO"])])
    mae_s = mae(y_test,model_predict_dt)
    rmse_s = rmse(y_test,model_predict_dt)
    mse_s = mse(y_test,model_predict_dt)
    r2_s = r2(y_test,model_predict_dt)
    print(best_score)
    print(r2_s)
    print(len(X))
    df_all_trees.append([best_model,best_score,mae_s,mse_s,rmse_s,r2_s,total_NaN_count_radiation,total_NaN_count,nome_usina,len(X)])
    print('%d modelos processados em aproximadamente %.2f segundos.' % (((len(dt_cv.cv_results_))),t_f))
    #trees = pd.DataFrame(models[1:],columns=models[0])
    #print(trees[trees['cv_s_m']>75].describe())
    grid = {
      'n_estimators': [1, 2, 5, 10, 15, 20],
      'max_features': [1.0],
      'max_depth' : [5,7,10,12,15,20],
      'criterion' :['squared_error'],
      'random_state' : [42]}
    tscv = TimeSeriesSplit()
    rf_cv = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv=tscv, scoring="r2", return_train_score=True)
    t_0 = time.time()
    rf_cv.fit(X_train, y_train)
    t_f = time.time() - t_0
    print("Fit_time =",t_f)
    forests = pd.DataFrame(rf_cv.cv_results_)
    print(forests)
    best_score = rf_cv.best_score_
    best_model = rf_cv.best_estimator_
    best_params = rf_cv.best_estimator_
    model_predict_rf = best_model.predict(X_test)
    if nome_usina in best_usinas:
      previsoes.append(["floresta aleatória",nome_usina,y_test,pd.DataFrame(model_predict_rf,index=y_test.index,columns=["GERACAO"])])
    mae_s = mae(y_test,model_predict_rf)
    rmse_s = rmse(y_test,model_predict_rf)
    mse_s = mse(y_test,model_predict_rf)
    r2_s = r2(y_test,model_predict_rf)
    print(best_score)
    print(r2_s)
    
    df_all_forests.append([best_model,best_score,mae_s,mse_s,rmse_s,r2_s,total_NaN_count_radiation,total_NaN_count,nome_usina,len(X)])
    #param_grid = {'n_estimators': [1, 2, 5,....], 'max_depth': [5, 10, .....]}
    print('%d modelos processados em aproximadamente %.2f segundos.' % (((len(rf_cv.cv_results_))),t_f))
    #df_all_forests.append([best_model,best_score,mae,mse,rmse,r2,total_NaN_count_radiation,total_NaN_count,nome_usina,len(X)])#Trocar pra ascending, n precisa, quanto maior melhor

trees_columns = ["model","score","mae","mse","rmse","r2","NaN_count_radiation","NaN_count_other","nome_usina","rows"]
df_all_trees = pd.DataFrame(df_all_trees,columns=trees_columns)
df_all_forests = pd.DataFrame(df_all_forests,columns=trees_columns)
tf_exec = time.time() - t0_exec
print("Tempo total de execução: %.2f minutos"%(tf_exec/60))
if plot1:
  xticks = []
  xtick_labels_text = []
  for item in max_xtick_labels:
    xticks.append(date2num(item))
    xtick_labels_text.append(str(item)[2:-12])
  usina_group = {}
  y_pos = np.arange(len(ranges))
  print(y_pos)
  fig, ax = plt.subplots()
  lines=[]
  for group in range(0,len(ranges)):
    usina_group[group] = str(group)
  for dados in plot_usinas_group.keys():
    start_date = plot_usinas_group[dados][0]
    end_date = plot_usinas_group[dados][1]
    group = plot_usinas_group[dados][2]
    print(group)
    indice_train = plot_usinas_group[dados][3]
    indice_test = plot_usinas_group[dados][4]

    lines+=ax.barh([group], (indice_train[-1] - indice_train[0]).days, left=indice_train[0], color='blue')
    lines+=ax.barh([group], (indice_test[-1] - indice_test[0]).days, left=indice_train[-1], color='orange')
  print(usina_group)
  print(ranges)
  plt.xticks(xticks, xtick_labels_text)
  plt.yticks(y_pos, usina_group.values())
  ax.legend(lines[:2],["Treino","Teste"],frameon = False,fontsize = "20")
  plt.xlabel("Intervalo de dados")
  plt.ylabel("Grupo")
  plt.title("Intervalo de Datas e Divisão Treino X Teste")
  plt.show()
if plot2:
  for contador in range(0,len(previsoes)):
    df1 = previsoes[contador][2]["2023-02-10 00:00:00":]
    df2 = previsoes[contador][3]["2023-02-10 00:00:00":]
    plt.title("Gráfico da "+previsoes[contador][0]+" da usina "+previsoes[contador][1])
    line1 = plt.plot(df1.index,df1.values,label="Real")
    line2 = plt.plot(df2.index,df2["GERACAO"],label="Previsto")
    plt.legend()
    plt.show()
