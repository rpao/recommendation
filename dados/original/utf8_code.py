arquivos = ['tags.csv', 'tips.csv']

for arquivo in arquivos:         
    ## Acessando arquivo
    ref_arquivo = open(arquivo, 'r')
    linha = ref_arquivo.readlines()
    ref_arquivo.close()
    for i in range (len(linha)):
        linha[i] = linha[i].strip()
        linha[i] = linha[i].decode('utf-8', 'ignore')
        linha[i] = linha[i].encode('utf-8', 'ignore')+"\n"
##        linha[i] = ([j if ord(j) < 128 else '' for j in linha[i]])[0]
    
    ## Escrevendo novo arqivo
    novo_arquivo = open('corrigido_'+arquivo, 'w')
    novo_arquivo.writelines(linha)
    novo_arquivo.close()
