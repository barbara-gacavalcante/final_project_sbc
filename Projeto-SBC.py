import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lime import lime_tabular
import streamlit.components.v1 as components



st.title("Previsão de Crédito")

@st.cache_data
def load_data(arquivo):
    return pd.read_csv(arquivo)

tabela = load_data("credit_data.csv")

st.write("### Dados dos Clientes:")
st.write(tabela.head(10))

codificador = LabelEncoder()

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

st.write(" ### Dados codificados:")
st.write(tabela.head(10))

# Dividir os dados
y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"], axis=1)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

explicador = lime_tabular.LimeTabularExplainer(
    x_treino.values,  # Dados de treinamento
    feature_names=x.columns,  # Nomes das colunas
    class_names=['Good', 'Poor', 'Standard'],  # Classes de saída
    verbose=True,
    mode='classification'
)

@st.cache_resource
def train_models():
    modelo_arvoredecisao = RandomForestClassifier()
    modelo_knn = KNeighborsClassifier()

    modelo_arvoredecisao.fit(x_treino, y_treino)
    modelo_knn.fit(x_treino, y_treino)

    return modelo_arvoredecisao, modelo_knn

modelo_arvoredecisao, modelo_knn = train_models()

st.write("### Precisão dos modelos")
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

# MOSTRANDO A ACURÁCIA DOS MODELOS EM DUAS COLUNAS BONITINHAS
col1, col2 = st.columns(2)
with col1:
    st.metric("###### Acurácia da árvore de decisão", value=accuracy_score(y_teste, previsao_arvoredecisao))

with col2:
    st.metric("###### Acurácia do modelo k-nearest neighbors", value=accuracy_score(y_teste, previsao_knn))

st.title("Novas previsões")
# PREVISÕES DOS NOVOS CLIENTE
with st.expander("Novas previsões"):
    novos_clientes = load_data("novos_clientes.csv")

    st.write("### Novos Dados dos Clientes:")
    st.write(novos_clientes)

    for coluna in novos_clientes.columns:
        if novos_clientes[coluna].dtype == "object" and coluna != "score_credito":
            novos_clientes[coluna] = codificador.fit_transform(novos_clientes[coluna])

    st.write(novos_clientes)

    previsao = modelo_arvoredecisao.predict(novos_clientes)
    st.write("### Novos Dados dos Clientes Codificados:")
    st.write(novos_clientes)

    colunas = st.columns(len(previsao))

    for i, resultado in enumerate(previsao):
        with colunas[i]:
            st.metric(f"Resultado {i+1}", value=resultado)

# TESTAR O CRÉDITO DE UM NOVO CLIENTE
st.title("Avaliar cliente")
with st.expander("Adicionar um novo cliente"):
    with st.form(key='adicionar_cliente'):
        mes = st.number_input("Mês", min_value=1, max_value=12, value=1)
        idade = st.number_input("Idade", min_value=18, max_value=100, value=18)
        profissao = st.selectbox('Profissão', ["empresario", "advogado", "professor", "cientista", "mecanico", 
                                            "medico", "musico", "gerente_midia", "jornalista", "escritor", 
                                            "desenvolvedor", "arquiteto", "gerente", "contador"])
        salario_anual = st.number_input("Salário Anual", min_value=0.0, value=20000.0)
        num_contas = st.number_input("Número de Contas", min_value=0, value=1)
        num_cartoes = st.number_input("Número de Cartões", min_value=0, value=1)
        juros_emprestimo = st.number_input("Juros Empréstimo (%)", min_value=0.0, value=10.0)
        num_emprestimos = st.number_input("Número de Empréstimos", min_value=0, value=1)
        dias_atraso = st.number_input("Dias de Atraso", min_value=0, value=0)
        num_pagamentos_atrasados = st.number_input("Número de Pagamentos Atrasados", min_value=0, value=0)
        num_verificacoes_credito = st.number_input("Número de Verificações de Crédito", min_value=0, value=0)
        mix_credito = st.selectbox("Mix de Crédito",["Bom", "Normal", "Ruim"])
        divida_total = st.number_input("Dívida Total", min_value=0.0, value=0.0)
        taxa_uso_credito = st.number_input("Taxa de Uso do Crédito (%)", min_value=0.0, value=0.0)
        idade_historico_credito = st.number_input("Idade do Histórico de Crédito (meses)", min_value=0, value=0)
        investimento_mensal = st.number_input("Investimento Mensal", min_value=0.0, value=0.0)
        comportamento_pagamento = st.selectbox("Comportamento de Pagamento", 
                                            ["baixo_gasto_pagamento_baixo", "baixo_gasto_pagamento_medio", 
                                                "baixo_gasto_pagamento_alto","alto_gasto_pagamento_baixos", 
                                                "alto_gasto_pagamento_medio", "alto_gasto_pagamento_alto"])
        saldo_final_mes = st.number_input("Saldo Final do Mês", min_value=0.0, value=0.0)
        emprestimo_carro = st.number_input("Emprestimo - Carro", min_value=0, max_value=1, value=0)
        emprestimo_casa = st.number_input("Emprestimo - Casa", min_value=0, max_value=1, value=0)
        emprestimo_pessoal = st.number_input("Emprestimo - Pessoal", min_value=0, max_value=1, value=0)
        emprestimo_credito = st.number_input("Emprestimo - Credito", min_value=0, max_value=1, value=0)
        emprestimo_estudantil = st.number_input("Emprestimo - Estudantil", min_value=0, max_value=1, value=0)
        submit_button = st.form_submit_button(label='Avaliar')

    if submit_button:
        dados_cliente = {
            'mes': mes,
            'idade': idade,
            'profissao': profissao,
            'salario_anual': salario_anual,
            'num_contas': num_contas,
            'num_cartoes': num_cartoes,
            'juros_emprestimo': juros_emprestimo,
            'num_emprestimos': num_emprestimos,
            'dias_atraso': dias_atraso,
            'num_pagamentos_atrasados': num_pagamentos_atrasados,
            'num_verificacoes_credito': num_verificacoes_credito,
            'mix_credito': mix_credito,
            'divida_total': divida_total,
            'taxa_uso_credito': taxa_uso_credito,
            'idade_historico_credito': idade_historico_credito,
            'investimento_mensal': investimento_mensal,
            'comportamento_pagamento': comportamento_pagamento,
            'saldo_final_mes': saldo_final_mes,
            'emprestimo_carro': emprestimo_carro,
            'emprestimo_casa': emprestimo_casa,
            'emprestimo_pessoal': emprestimo_pessoal,
            'emprestimo_credito': emprestimo_credito,
            'emprestimo_estudantil': emprestimo_estudantil
        }


        df_novo_cliente = pd.DataFrame([dados_cliente])
        for coluna in df_novo_cliente.columns:
            if df_novo_cliente[coluna].dtype == "object" and coluna != "score_credito":
                df_novo_cliente[coluna] = codificador.fit_transform(df_novo_cliente[coluna])
        
        st.write("### Avaliação do Cliente:")
        st.write(df_novo_cliente)
        
        previsao = modelo_arvoredecisao.predict(df_novo_cliente)

        st.metric(label="Previsão de Score de Crédito", value=previsao[0])

        # Gerar explicação usando LIME
        explicacao = explicador.explain_instance(df_novo_cliente.values[0], modelo_arvoredecisao.predict_proba, num_features = 20, num_samples=10000)

        st.write("### Explicação da Previsão com LIME:")
        components.html(explicacao.as_html(), height=500)