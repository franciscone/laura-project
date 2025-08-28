import streamlit as st
from docling.document_converter import DocumentConverter
from huggingface_hub import InferenceClient
import os

## load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_API_TOKEN")
)

st.set_page_config(page_title="Chat com Documentos", layout="wide")
st.title("📄 Robô Auditor SESA - Módulo MLops")

st.chat_message("assistant").markdown("Olá, insira a documentação necessária para que eu possa resumir e listar os principais pontos para você!")

uploaded_files = st.file_uploader(
    "Upload data", accept_multiple_files=True, type="pdf"
)

if uploaded_files:
    texts = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        
        converter = DocumentConverter()
        doc = converter.convert(file.name).document

        texts.append(doc.export_to_markdown())


    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
                            Você é um robô auditor da SESA. Sua tarefa é ler e analisar um ou mais documentos apresentados.  
                            ***Sempre*** responda no seguinte formato fixo em Markdown:

                            # 📄 Análise do(s) Documento(s)

                            ## ✅ O que aprendi
                            Liste em tópicos os principais aprendizados de cada documento, de forma clara e objetiva.

                            ## 🔍 Comparação entre documentos
                            Se houver mais de um documento:
                            - Similaridades
                            - Diferenças
                            - Lacunas
                            - Redundâncias
                            - Complementaridades

                            ## 🏥 Como pode ser funcional
                            Explique como os aprendizados podem ser úteis em diferentes áreas da SESA, e como podem apoiar decisões e operações.

                            ## ❓ Perguntas que consigo responder
                            Liste exemplos de perguntas que você poderia responder com base nos documentos.

                            ## 📊 Métricas do aprendizado
                            - **Nível de confiança**: 1-10
                            - **Tipo de conteúdo identificado**: ex.: Lei, Licitação, Relatório, Outro
                            - **Métricas adicionais**: adicione aqui qualquer métrica relevante dependendo do documento (ex.: quantidade de leis, número de artigos, quantidade de licitações, páginas de relatório, valores envolvidos, anos de publicação, órgãos envolvidos, indicadores, etc.)

                            ---
                            ⚠️ Sempre use o Markdown para visualização.
                            """

            }
            ,{
                "role": "user",
                "content": "\n\n---\n\n".join(texts)
            }
        ],
        max_tokens=150000
    )

    st.chat_message("assistant").markdown(completion.choices[0].message.content)

    

