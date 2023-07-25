import pandas as pd
import openpyxl
from difflib import get_close_matches
import os
import csv
#Dados dos nomes de usinas
def load_nomes_usinas():
    df=pd.read_csv("USINAS BR.csv",header=0,index_col=0)
    #print(df)
    df_solar_names = df[df["SigTipoGeracao"]=="UFV"]
    #df_solar_names = df_solar_names[df_solar_names["DscFaseUsina"]=="Operação"]
    #df_solar_names = df_solar_names[df_solar_names["DatEntradaOperacao"]<"2014"]
    names = df_solar_names["DscMuninicpios"].to_list()
    #print(names)
    return(df_solar_names)

def import_met_locations():
    cwd = os.getcwd()
    # list all files in current directory
    file_list = os.listdir(cwd+r'\\climatico')[1:]
    dic ={}

    for file in file_list:
        if ".py" not in file:
            with open(r'climatico\\'+file, 'r') as f:
                # Create a CSV reader object
                csv_reader = csv.reader(f)
                csv_reader = list(csv_reader)
                # Loop through each row in the CSV file
                #print(csv_reader[0:7])
                if len(csv_reader[4])==2:
                    lat = csv_reader[4][0].split(";")[1]+","+csv_reader[4][1]
                else:
                    lat = csv_reader[4][0].split(";")[1]
                if len(csv_reader[5])==2:
                    long = csv_reader[5][0].split(";")[1]+","+csv_reader[5][1]
                else:
                    long = csv_reader[5][0].split(";")[1]
                estacao = csv_reader[2][0].split(";")[1]
                #print(estacao,lat,long)
                #estacao == nome da cidade
                dic[estacao]=[lat,long]
    return dic
def pegar_relacoes():
    file = "tabela_relacao_usinas.csv"
    with open(file, 'r') as f:
        # Create a CSV reader object
        csv_reader = csv.reader(x.replace('\0', '') for x in f)
        csv_reader = list(csv_reader)
        output = []
        #print('\t')
        for line in csv_reader:
            if line == []:
                continue
            else:
                #print(line[5].split('\t')[3])
                lista = line[4].split('\t')
                lista.append(line[5].split('\t')[3][:-2]+line[5].split('\t')[3][-1:])
                lista[5] = lista[5][4:]
                output.append(lista)
    df = pd.DataFrame(output,columns=["asd","Regiao","Estado","das","sda","nome_geracao","sad","nome_ceg"])
    df.drop(columns=["asd","das","sda","sad","Regiao","Estado"],inplace=True)
    nome_geracao = df["nome_geracao"].to_list()
    nome_geracao_upper=[]
    for item in nome_geracao:
        nome_geracao_upper.append(item.upper())
    nome_ceg = df["nome_ceg"].to_list()
    return df
def load_geracao_usinas():
    cwd = os.getcwd()
    # list all files in current directory
    file_list = os.listdir(cwd+r'\\geracao')
    dic ={}
    usinas_solares = pd.DataFrame()
    #print(file_list)
    for file in file_list:
        df=pd.read_excel(r'geracao\\'+file,sheet_name=8,header=14,index_col=1)
        df = df[df["Fonte"]=="Solar Fotovoltaica"]
        #usinas_solares = usinas_solares[~usinas_solares["Sigla da Usina"].duplicated(keep=False)]
        #print(usinas_solares.columns)
        df = df[["Sigla da Usina","Hora","Dia","Geração no Centro de Gravidade - MW médios (Gp,j) - MWh"]]
        #print(usinas_solares)
        #print(type(usinas_solares.value_counts("Sigla da Usina")))#[usinas_solares["Sigla da Usina"]=="TAUA"]
        usinas_solares = usinas_solares.append(df,ignore_index=True)
    return(usinas_solares)
def match_names(name, choices):
    matches = get_close_matches(name, choices, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    else:
        return None
cwd = os.getcwd()
geracao_folder = cwd+r'\\geracao_csv\\'
df_correl = pegar_relacoes()
df_nomes = load_nomes_usinas()
df_geracao = load_geracao_usinas()

print(df_geracao.nunique())
# sample dataframes

# function to match names in two columns
#print(df_geracao['Sigla da Usina'])
df_geracao["nomes_match"] = df_geracao['Sigla da Usina'].apply(lambda x: x.upper())
print(df_geracao)
df_correl["nome_upper"] = df_correl["nome_geracao"].apply(lambda x: x.upper())
#print(df_nomes["CodCEG"])
#print(df_correl["nome_ceg"])

# merge df1 with df2 based on matched_names column
#merged_df = pd.merge(df_geracao, df_correl, left_on='nomes_match', right_on='nome_geracao', how='inner')
merged_df = pd.merge(df_geracao, df_correl, left_on='nomes_match', right_on='nome_upper', how='inner')
print(merged_df)
merged_df = pd.merge(merged_df, df_nomes, left_on='nome_ceg', right_on='CodCEG', how='inner')
print(merged_df)
lista_usinas = (merged_df["nomes_match"].unique())
print(lista_usinas)

#lista_usinas = ["COREMAS III","BOA HORA 1","VAZANTE 1","SOL DO FUTURO III","PIRAPORA 5"]
#merged_df = merged_df[merged_df["Sigla da Usina"].str.contains("COREMAS III|BOA HORA 1|VAZANTE 1|SOL DO FUTURO III|PIRAPORA 5")]
#merged_df.to_csv("dados_geracao.csv")
for usina in lista_usinas:
    out_df = merged_df[merged_df["nomes_match"] == usina].sort_values("Dia")
    out_df.to_csv(geracao_folder+usina+"_geracao.csv")
#print(merged_df)
