import unicodedata
import re
import numpy as np
from skimage.filters import threshold_otsu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RelevanciaModel:
    def preprocess_text(self, text: str) -> str:
        """Remove acentos e pontuação do texto e retorna-o em minúsculas."""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def get_embeddings(self, texts, model):
        """Gera embeddings para uma lista de textos usando o modelo fornecido."""
        return model.encode(texts)

    def compute_similarity(self, embedding1, embedding2) -> float:
        """Calcula a similaridade de cosseno entre dois embeddings e converte para float nativo."""
        return float(cosine_similarity([embedding1], [embedding2])[0][0])


def auto_threshold(similarity_scores):
    """
    Calcula automaticamente um limiar utilizando o método de Otsu com base na distribuição dos escores.
    """
    sim_array = np.array(similarity_scores)
    threshold = threshold_otsu(sim_array)
    return float(threshold)


class AnalisadorRelevancia:
    def __init__(self):
        # Carrega um modelo multilíngue pré-treinado.
        self.modelo = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.model = RelevanciaModel()
        self.historico = []  # Armazena os resultados de cada avaliação.
        self.limiar_padrao = 0.45  # Limiar padrão utilizado.

    def avaliar_relevancia(self, pergunta: str, resposta: str) -> dict:
        """
        Recebe a pergunta e a resposta, realiza o pré-processamento, gera os embeddings,
        calcula a similaridade e armazena o resultado usando o limiar padrão.
        """
        pergunta_proc = self.model.preprocess_text(pergunta)
        resposta_proc = self.model.preprocess_text(resposta)

        embeddings = self.model.get_embeddings([pergunta_proc, resposta_proc], self.modelo)
        similaridade = self.model.compute_similarity(embeddings[0], embeddings[1])
        relevante = similaridade >= self.limiar_padrao

        resultado = {
            'pergunta': pergunta,
            'resposta': resposta,
            'similaridade': similaridade,
            'limiar': float(self.limiar_padrao),
            'relevante': bool(relevante)
        }
        self.historico.append(resultado)
        return resultado

    def _interpretar_score(self, score: float) -> dict:
        """Retorna uma análise textual do nível de similaridade."""
        if score >= 0.7:
            nivel = "Excelente correspondência"
            observacoes = [
                "Relação temática muito forte",
                "Contextos perfeitamente alinhados"
            ]
        elif score >= 0.5:
            nivel = "Boa correspondência"
            observacoes = [
                "Tópico principal abordado",
                "Alguns pontos relevantes presentes"
            ]
        elif score >= 0.3:
            nivel = "Correspondência parcial"
            observacoes = [
                "Relação superficial com o tema",
                "Faltaram elementos essenciais"
            ]
        else:
            nivel = "Sem correspondência relevante"
            observacoes = [
                "Tópicos completamente divergentes",
                "Necessária revisão completa"
            ]
        return {"nivel": nivel, "observacoes": observacoes}

    def gerar_relatorio(self, indice: int = -1) -> dict:
        """
        Gera um relatório detalhado para o resultado especificado (última avaliação por padrão),
        retornando um JSON com a estrutura aprimorada.
        """
        dados = self.historico[indice]
        status_text = "Relevante" if dados['relevante'] else "Não relevante"

        detalhes = {
            "valorSimilaridade": dados['similaridade'],
            "limiarUtilizado": dados['limiar'],
            "status": status_text
        }
        analise = self._interpretar_score(dados['similaridade'])

        if dados['relevante']:
            recomendacoes = [
                "A resposta aborda adequadamente o tema da pergunta",
                "Conteúdo mantém coerência com o contexto solicitado"
            ]
        else:
            recomendacoes = [
                "Revisar o alinhamento temático da resposta",
                "Incluir elementos essenciais que abordem a pergunta de forma completa"
            ]

        relatorio = {
            "pergunta": dados['pergunta'],
            "resposta": dados['resposta'],
            "similaridade": dados['similaridade'],
            "limiar": dados['limiar'],
            "relevante": dados['relevante'],
            "detalhes": detalhes,
            "analise": analise,
            "recomendacoes": recomendacoes
        }
        return relatorio

    def ajustar_limiar_automaticamente(self) -> float:
        """
        Recalcula e retorna um novo limiar com base nos escores do histórico usando o método de Otsu.
        """
        scores = [d['similaridade'] for d in self.historico]
        if len(scores) > 1:
            novo_limiar = auto_threshold(scores)
            self.limiar_padrao = novo_limiar
            return novo_limiar
        return float(self.limiar_padrao)


class RelevanciaController:
    """
    Controller que encapsula a lógica de avaliação de relevância.
    """

    def __init__(self):
        self.analisador = AnalisadorRelevancia()

    def processar_consulta(self, pergunta: str, resposta: str) -> dict:
        """Processa a consulta e retorna o resultado da avaliação."""
        return self.analisador.avaliar_relevancia(pergunta, resposta)

    def exibir_relatorio(self, indice: int = -1) -> dict:
        """Retorna o relatório detalhado da consulta especificada."""
        return self.analisador.gerar_relatorio(indice)

    def ajustar_limiar(self) -> float:
        """Ajusta automaticamente o limiar com base no histórico e retorna o novo valor."""
        return self.analisador.ajustar_limiar_automaticamente()

    def obter_historico(self) -> list:
        """Retorna o histórico completo de avaliações."""
        return self.analisador.historico
