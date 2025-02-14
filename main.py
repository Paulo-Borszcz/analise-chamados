import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.preprocessing import normalize
import plotly.graph_objects as go

# Baixar recursos do NLTK (caso ainda não estejam disponíveis)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Carregar o arquivo CSV e verificar as colunas necessárias
try:
    data = pd.read_csv('data.csv')
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    exit()

required_columns = ['id', 'content', 'category_name']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Coluna '{col}' não encontrada no CSV.")

# Função para limpar o conteúdo: remove HTML, pontuação e stopwords
def clean_text(text):
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text().lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('portuguese'))
    filtered = [token for token in tokens if token not in stop_words]
    return " ".join(filtered)

print("Limpando textos...")
data['cleaned_content'] = data['content'].apply(clean_text)
print("Limpeza concluída.")

# Gerar embeddings usando um modelo avançado
print("Gerando embeddings com all-mpnet-base-v2...")
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(data['cleaned_content'].tolist(), show_progress_bar=True)

# Normalizar os embeddings e reduzir para 3D com UMAP
print("Reduzindo dimensionalidade para 3D com UMAP...")
embeddings_norm = normalize(embeddings)
umap_model = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine')
reduced = umap_model.fit_transform(embeddings_norm)
data['umap_x'] = reduced[:, 0]
data['umap_y'] = reduced[:, 1]
data['umap_z'] = reduced[:, 2]

# Gerar traços (traces) para cada categoria
categories = sorted(data['category_name'].unique())
traces = []
for cat in categories:
    df_cat = data[data['category_name'] == cat]
    trace = go.Scatter3d(
        x = df_cat['umap_x'],
        y = df_cat['umap_y'],
        z = df_cat['umap_z'],
        mode = "markers",
        marker = dict(size=6, symbol="circle"),
        name = cat,
        text = df_cat['id'].astype(str),
        hovertemplate = "<b>ID</b>: %{text}<br>" +
                        "<b>Categoria</b>: " + cat + "<br>" +
                        "UMAP X: %{x:.3f}<br>UMAP Y: %{y:.3f}<br>UMAP Z: %{z:.3f}<extra></extra>"
    )
    traces.append(trace)

# Criar figura com os traços
fig = go.Figure(data=traces)

# Criar botões do menu dropdown: um para "Todos" e um para cada categoria
buttons = []
# Botão para mostrar todos os traços
buttons.append(dict(
    label="Todos",
    method="update",
    args=[{"visible": [True] * len(traces)},
          {"title": "3D Scatter Plot: Todas as Categorias"}]
))
# Botões para cada categoria individual
for i, cat in enumerate(categories):
    visibility = [False] * len(traces)
    visibility[i] = True
    buttons.append(dict(
        label=cat,
        method="update",
        args=[{"visible": visibility},
              {"title": f"3D Scatter Plot: {cat}"}]
    ))

# Atualizar o layout: posiciona o menu dropdown no canto superior esquerdo
fig.update_layout(
    title="3D Scatter Plot: Todas as Categorias",
    scene=dict(
        xaxis_title="UMAP X",
        yaxis_title="UMAP Y",
        zaxis_title="UMAP Z"
    ),
    template="plotly_white",
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.02,
        xanchor="left",
        y=1.15,
        yanchor="top"
    )]
)

# Salvar o gráfico interativo em um HTML estático
output_file = 'interactive_3d_umap_chamados_filtered.html'
fig.write_html(output_file, include_plotlyjs='cdn')
print(f"Gráfico salvo em '{output_file}'. Você pode hospedar esse arquivo HTML para acessar a visualização interativa.")
