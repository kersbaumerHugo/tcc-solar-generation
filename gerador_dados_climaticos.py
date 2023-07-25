import pandas as pd
import openpyxl
import os
import csv
from geopy import distance
import time as t
def open_geracao():
    cwd = os.getcwd()
    file_list = os.listdir(cwd+r'\\geracao_csv\\')
    all_geracao = []
    coords = {}
    for file in file_list:
        if "geracao.csv" in file:
            with open(cwd+r'\\geracao_csv\\'+file, 'r',encoding="utf-8") as f:
                # Create a CSV reader object
                csv_reader = csv.reader(f)
                csv_reader = list(csv_reader)
                for line in csv_reader[1:]:
                    name_and_state = line[1]+" "+line[13]
                    lat,long = line[25].replace(",","."),line[26].replace(",",".")
                    if name_and_state not in coords.keys():
                        coords[name_and_state] = [lat,long]
                        
    return coords
def calculate_dist(list1,list2):

    coord1 = (list1[0],list1[1])  # New York City coordinates
    coord2 = (list2[0],list2[1])   # London coordinates

    dist = distance.distance(coord1, coord2).km  # calculate distance in kilometers

    return(dist)
def preparar_merge(df,contador):
    #print(df.columns)
    df["Hora UTC"+str(contador)] = df["Hora UTC"+str(contador)].apply(lambda x: x[:2]+":"+x[2:-4]+":"+"00")
    df["Date"+str(contador)] = df["Data"+str(contador)]+" "+df["Hora UTC"+str(contador)]
    df["Date"+str(contador)] = pd.to_datetime(df["Date"+str(contador)])
    df.set_index("Date"+str(contador),inplace=True)
    df = df.drop(columns=["Data"+str(contador),"Hora UTC"+str(contador)])
    return df
def import_met_locations(year,recover_data=False,id_estacao="",position=0):
    cwd = os.getcwd()
    # list all files in current directory
    file_list = os.listdir(cwd+r'\\climatico\\'+year)
    dic ={}
    for file in file_list:
        if ".py" not in file:
            with open(r'climatico\\'+year+r'\\'+file, 'r') as f:
                # Create a CSV reader object
                csv_reader = csv.reader(f,delimiter=";")
                csv_reader = list(csv_reader)
                # Loop through each row in the CSV file
                #print(csv_reader[0:7])
                if recover_data:
                    if csv_reader[3][1] == id_estacao:
                        headers_no_position = csv_reader[8]
                        headers = []
                        for header in headers_no_position:
                            header+=str(position)
                            #print(header)
                            headers.append(header)
                        #headers = headers[:-1]
                        data = csv_reader[9:]
                        df = pd.DataFrame(data)
                        df.columns = headers
                        df.drop(columns=[str(position)])
                        #print(df)
                else:
                    lat = csv_reader[4][1]
                    long = csv_reader[5][1]
                    estacao = csv_reader[3][1]
                    #print(estacao,lat,long)
                    #estacao == nome da cidade
                    dic[estacao]=[lat.replace(",","."),long.replace(",",".")]
    if recover_data:
        #print(df)
        return df
    else:
        return dic
dic = import_met_locations("2023")
plant_coords = open_geracao()
city_list = {}
for id_city,coord in plant_coords.items():
    #city_list[id_city] = ["",500]#0 é o header
    station_list = []
    for key,value in dic.items():
        dist_calc = calculate_dist(dic[key],coord)
        station_list.append([key,dist_calc])
    station_df = pd.DataFrame(station_list,columns=["Estacao","Distancia"])
    station_df = station_df.sort_values("Distancia",ignore_index=True)[:3]
    city_list[id_city] = []
    for row in station_df.itertuples():
     city_list[id_city].append([row[1],row[2]])
#print(city_list)


cwd = os.getcwd()
# list all files in current directory
file_list = os.listdir(cwd+r'\\climatico\\')
#print(file_list)
dic ={}
total = len(city_list.keys())
start_time = t.time()
for key in city_list.keys():
    porcentagem = 0
    contador = 0
    df_saida = import_met_locations("2023",recover_data=True,id_estacao=city_list[key][0][0],position=contador)
    df_saida = preparar_merge(df_saida,contador)
    contador+=1
    for station in city_list[key][1:]:
        #print(contador)
        df_station = import_met_locations("2023",recover_data=True,id_estacao=city_list[key][contador][0],position=contador)
        df_station = preparar_merge(df_station,contador)
        #print(df_station)
        #print(df_saida)
        size_antes = len(df_saida.index)
        df_saida = df_saida.merge(df_station,left_index=True,right_index=True)
        #print(len(df_saida.index)==size_antes)
        #print(df_saida)
        contador+=1
    for year in file_list[:-1]:
        contador = 0
        
        df_all_stations = import_met_locations(year,recover_data=True,id_estacao=city_list[key][0][0],position=contador)
        df_all_stations = preparar_merge(df_all_stations,contador)
        contador+=1
        for station in city_list[key][1:]:
            df_station = import_met_locations(year,recover_data=True,id_estacao=city_list[key][contador][0],position=contador)
            df_station = preparar_merge(df_station,contador)
            size_antes = len(df_saida.index)
            df_all_stations = df_all_stations.merge(df_station,left_index=True,right_index=True)
            #print(len(df_saida.index)==size_antes)
            contador+=1
        #print(len(df_saida.index))
        #print(df_saida)
        df_saida = pd.concat([df_all_stations,df_saida])
        #print(len(df_saida.index))
    porcentagem+=1
    print("Processado = %.2f"%(porcentagem/total))
    print(t.time()-start_time)
        #print(df_saida)
    #Nome da estação:
    #Qual inicio e final da série?
    #Quantos zeros?
    #Quantos N/A nas colunas?
    df_saida.to_csv(cwd+r'clima_csv'+key[:-3].upper()+"_clima.csv")        
