
"""
# Permitir entrada de novos dados para previsão
st.write("### Previsão para Novos Clientes")

# Input de novos clientes (pode ajustar as entradas de acordo com seu dataset)
mes = st.number_input('Mês', min_value=1, max_value=12, value=1)
idade = st.number_input('Idade', min_value=18, max_value=100, value=30)
profissao = st.selectbox('Profissão', [0, 1])
salario_anual = st.number_input('Salário Anual', min_value=0.0, value=20000.0)
num_contas = st.number_input('Número de Contas', min_value=0, max_value=10, value=5)

# (Adicione inputs para todas as colunas necessárias para o modelo)

# Agrupar entradas em um DataFrame
novos_dados = pd.DataFrame({
    'mes': [mes],
    'idade': [idade],
    'profissao': [profissao],
    'salario_anual': [salario_anual],
    'num_contas': [num_contas],
    # Adicionar os outros campos
})

# Exibir dados de entrada
st.write("### Dados de Entrada:")
st.write(novos_dados)

# Fazer a previsão
previsao = modelo_arvoredecisao.predict(novos_dados)

st.write("### Previsão:")
st.write(previsao)
"""