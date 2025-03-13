# API Juiz LLM

API para validação de respostas geradas por modelos de linguagem (LLMs) utilizando IA Constitucional e métricas de avaliação baseadas em Ragas.

## 📋 Descrição

Este projeto implementa um sistema de validação de respostas de LLMs que combina dois enfoques complementares:

1. **IA Constitucional**  
   Inspirada na abordagem da Anthropic, essa camada (representada pelo módulo de conformidade – *Compliance*) visa garantir que as respostas estejam alinhadas com diretrizes éticas e constitucionais pré-definidas.  
   - Utiliza modelos pré-treinados (BERT para embeddings) e técnicas de NLP para detectar violação de princípios constitucionais (ex.: privacidade, ética).
   - Realiza análise de termos críticos, stemming e verificação semântica para identificar possíveis violações.
   - Explica as decisões por meio de relatórios detalhados e métricas de risco.

2. **Métricas Ragas**  
   Baseado no framework Ragas, essa camada (representada pelo módulo de relevância – *Relevancia*) mede a qualidade técnica e contextual da resposta:
   - Compara a similaridade entre a pergunta e a resposta utilizando embeddings gerados pelo modelo [SentenceTransformers](https://www.sbert.net/).
   - Calcula métricas como similaridade de cosseno, interpretando o resultado em níveis (ex.: excelente, boa, parcial ou sem correspondência).
   - Permite ajustar automaticamente o limiar para classificação de relevância.

A validação ocorre em três etapas:
1. **Filtragem Constitucional**: Análise de conformidade utilizando o módulo de *Compliance*.
2. **Medição de Métricas**: Cálculo de similaridade entre pergunta e resposta com base na abordagem de *Relevancia*.
3. **Análise Contextual e Inferência**: Verificação de consistência factual, análise de relevância e detecção de alucinações.

## ✨ Funcionalidades

- **Validação Constitucional (IA Constitucional)**:
  - Classificação de respostas quanto ao cumprimento de diretrizes éticas e legais.
  - Detecção de termos críticos e violações (ex.: privacidade, ética).
  - Geração de relatórios detalhados com motivos, risco e detalhes da violação.

- **Métricas Ragas para Relevância**:
  - Cálculo de similaridade de cosseno entre pergunta e resposta.
  - Análise qualitativa (nível de correspondência e recomendações de melhoria).
  - Ajuste automático de limiar com base na distribuição dos escores.

- **API REST**:
  - Endpoints para avaliação de relevância e conformidade.
  - Documentação interativa via Swagger/Redoc (acessível em `/docs` ou `/redoc`).
  - Suporte para consulta em batch e histórico de avaliações.

## 🚀 Endpoints Principais

- **Relevância**:
  - `POST /avaliar`: Recebe um JSON com os campos `"pergunta"` e `"resposta"`, processa a análise de relevância e retorna um relatório estruturado.
  - `GET /historico`: Retorna o histórico completo das avaliações de relevância.
  - `GET /ajustar-limiar`: Ajusta automaticamente o limiar com base no histórico e retorna o novo valor.

- **Conformidade (IA Constitucional)**:
  - `POST /analisar`: Recebe um JSON com o campo `"texto"` e executa a análise de conformidade, retornando os resultados em formato JSON.
  - `GET /version`: Retorna a versão do sistema de análise de conformidade.

## 🛠️ Tecnologias e Dependências

- **Linguagem**: Python 3.10+
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Modelos e NLP**:
  - [SentenceTransformers](https://www.sbert.net/) para análise de relevância.
  - [Transformers](https://huggingface.co/transformers/) (BERT) para a análise de conformidade.
- **Bibliotecas**:
  - `torch`
  - `numpy`
  - `scikit-learn`
  - `scikit-image`
  - `nltk`
  - `transformers`
  - `sentence-transformers`
- **Logging**: Configurado para registrar logs em `compliance_analysis.log`.

## ⚙️ Instalação

### Pré-requisitos

- Certifique-se de ter o Python 3.10 ou superior instalado.
- Recomenda-se criar um ambiente virtual para o projeto:

```bash
# Crie e ative o ambiente virtual (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

# No Windows
python -m venv venv
venv\Scripts\activate
