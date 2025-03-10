import unicodedata
import re
import numpy as np
from skimage.filters import threshold_otsu  # pip install scikit-image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RelevanciaModel:
    def preprocess_text(self, text: str) -> str:
        """Remove acentos e pontuação do texto, retornando-o em minúsculas."""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def get_embeddings(self, texts, model):
        """Gera embeddings para uma lista de textos usando o modelo fornecido."""
        return model.encode(texts)

    def compute_similarity(self, embedding1, embedding2) -> float:
        """Calcula a similaridade de cosseno entre dois embeddings."""
        return cosine_similarity([embedding1], [embedding2])[0][0]


def auto_threshold(similarity_scores):
    """
    Calcula automaticamente um limiar utilizando o método de Otsu com base na distribuição dos escores.
    """
    sim_array = np.array(similarity_scores)
    threshold = threshold_otsu(sim_array)
    return threshold


class AnalisadorRelevancia:
    def __init__(self):
        # Carrega o modelo pré-treinado
        self.modelo = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.model = RelevanciaModel()
        self.historico = []  # Armazena os resultados de cada avaliação.
        self.limiar_padrao = 0.3  # Limiar padrão utilizado

    def avaliar_relevancia(self, pergunta: str, resposta: str) -> dict:
        """
        Recebe a pergunta e a resposta, realiza o pré-processamento, gera embeddings,
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
            'limiar': self.limiar_padrao,
            'relevante': relevante
        }
        self.historico.append(resultado)
        return resultado

    def _interpretar_score(self, score: float) -> str:
        """Retorna uma análise textual do nível de similaridade."""
        analise = "ANÁLISE DE RELEVÂNCIA:\n"
        if score >= 0.7:
            analise += "  Nível: Excelente correspondência\n"
            analise += "  - Relação temática muito forte\n"
            analise += "  - Contextos perfeitamente alinhados\n"
        elif score >= 0.5:
            analise += "  Nível: Boa correspondência\n"
            analise += "  - Tópico principal abordado\n"
            analise += "  - Alguns pontos relevantes presentes\n"
        elif score >= 0.3:
            analise += "  Nível: Correspondência parcial\n"
            analise += "  - Relação superficial com o tema\n"
            analise += "  - Faltaram elementos essenciais\n"
        else:
            analise += "  Nível: Sem correspondência relevante\n"
            analise += "  - Tópicos completamente divergentes\n"
            analise += "  - Necessária revisão completa\n"
        return analise

    def gerar_relatorio(self, indice: int = -1) -> dict:
        """
        Gera um relatório detalhado para o resultado especificado (última avaliação por padrão).
        """
        dados = self.historico[indice]
        relatorio = "\n" + "=" * 60 + "\n"
        relatorio += " RELATÓRIO DE RELEVÂNCIA SEMÂNTICA ".center(60, '#') + "\n"
        relatorio += "=" * 60 + "\n\n"
        relatorio += f"Pergunta: {dados['pergunta']}\n"
        relatorio += f"Resposta: {dados['resposta']}\n\n"
        relatorio += f"Similaridade: {dados['similaridade']:.2f}\n"
        relatorio += f"Limiar: {dados['limiar']:.2f}\n"
        relatorio += f"Status: {'RELEVANTE ' if dados['relevante'] else 'NÃO RELEVANTE '}\n\n"
        relatorio += "-" * 60 + "\n"
        relatorio += self._interpretar_score(dados['similaridade']) + "\n"
        relatorio += "-" * 60 + "\n"
        relatorio += "RECOMENDAÇÕES:\n"
        if dados['relevante']:
            relatorio += "  - A resposta aborda adequadamente o tema da pergunta\n"
            relatorio += "  - Conteúdo mantém coerência com o contexto solicitado\n"
        else:
            relatorio += "  - Resposta apresenta desvio temático significativo\n"
            relatorio += "  - Recomenda-se revisão para melhor alinhamento com a pergunta\n"
        relatorio += "=" * 60 + "\n"
        dados['relatorio'] = relatorio
        return dados

    def ajustar_limiar_automaticamente(self) -> float:
        """
        Recalcula e retorna um novo limiar com base nos escores do histórico usando o método de Otsu.
        """
        scores = [d['similaridade'] for d in self.historico]
        if len(scores) > 1:
            novo_limiar = auto_threshold(scores)
            self.limiar_padrao = novo_limiar
            return novo_limiar
        return self.limiar_padrao


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
