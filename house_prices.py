import pandas as pd
import numpy as np

# Definindo a semente aleatória para reprodutibilidade
np.random.seed(0)

# Gerando dados aleatórios
area = np.random.normal(150, 50, 2000)  # Distribuição normal com média 150 e desvio padrão 50
quartos = np.random.randint(1, 6, 2000)  # Valores inteiros aleatórios entre 1 e 5
banheiros = np.random.randint(1, 4, 2000)  # Valores inteiros aleatórios entre 1 e 3

# Suponhamos que o preço seja influenciado linearmente por estas características, com algum ruído aleatório
preco = 2000 + 300*area + 10000*quartos + 7000*banheiros + np.random.normal(0, 5000, 2000)

# Criando um DataFrame
df = pd.DataFrame({'Área': area, 'Quartos': quartos, 'Banheiros': banheiros, 'Preço': preco})

# Arredondando os preços para 2 casas decimais
df['Preço'] = df['Preço'].round(2)

# Salvando o DataFrame como um arquivo CSV
df.to_csv('house_prices.csv', index=False)