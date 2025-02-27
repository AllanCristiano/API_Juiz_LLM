# API Juiz LLM

API para validação de respostas geradas por modelos de linguagem (LLMs) utilizando IA Constitucional e métricas de avaliação baseadas em Ragas.

## 📋 Descrição

Este projeto implementa um sistema de validação de respostas de LLMs combinando dois enfoques:
1. **IA Constitucional** (inspirada na abordagem da Anthropic) para garantir alinhamento ético e diretrizes pré-definidas.
2. **Métricas de avaliação** baseadas no framework Ragas para análise de qualidade técnica.

A validação ocorre em três etapas:
1. Filtragem constitucional das respostas
2. Medição de métricas de qualidade
3. Análise de contexto e relevância

## ✨ Funcionalidades

- **Validação Constitucional**: 
  - Classificação de respostas usando modelo BART fine-tuned
  - Filtragem de conteúdo inadequado ou não-ético
  - Explicabilidade das decisões

- **Métricas Ragas**:
  - Context Precision (Precisão de Contexto)
  - Answer Correctness (Correção da Resposta)
  - Faithfulness (Fidelidade às Fontes)

- **Inferência Contextual**:
  - Análise de relevância resposta-pergunta
  - Verificação de consistência factual
  - Detecção de alucinações

- **API REST**:
  - Endpoint único para validação completa
  - Documentação interativa (Swagger/Redoc)
  - Escalável com suporte a batch processing
