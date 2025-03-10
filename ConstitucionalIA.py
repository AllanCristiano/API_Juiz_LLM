import torch
import numpy as np
import re
import unicodedata
import logging
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
import nltk

# Configuração do logging
logging.basicConfig(
    filename='compliance_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

nltk.download('rslp', quiet=True)


def _normalize_text(text):
    """Normaliza texto removendo acentos e caracteres especiais"""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'(?<!\w)[^\w\s!?](?!\w)', '', text)
    return text.strip()


class ComplianceResult:
    """Armazena e formata resultados da análise de conformidade"""

    def __init__(self, text, version):
        self.text = text
        self.version = version
        self.categories = {}
        self.violations = {}
        self.compliance_status = True
        self.error = None

    def add_category_result(self, category, similarity, threshold, detected_terms, violation):
        """Adiciona resultados de uma categoria específica"""
        self.categories[category] = {
            'similarity_score': float(round(similarity, 2)),
            'similarity_threshold': float(threshold),
            'detected_terms': detected_terms,
            'violation_triggered': violation
        }

        if violation:
            self.violations[category] = {
                'reasons': self._get_violation_reasons(similarity, threshold, detected_terms),
                'risk_score': self._calculate_risk_score(similarity, detected_terms),
                'details': self._get_violation_details(category, detected_terms, similarity, threshold)
            }
            self.compliance_status = False

    def _get_violation_reasons(self, similarity, threshold, detected_terms):
        """Identifica os motivos da violação"""
        reasons = []
        if similarity >= threshold * 1.1:
            reasons.append('similaridade_alta')
        if detected_terms and similarity >= threshold * 0.8:
            reasons.append('combinacao_termos_similaridade')
        return reasons

    def _calculate_risk_score(self, similarity, detected_terms):
        """Calcula o score de risco normalizado"""
        term_factor = 0.4 if any(detected_terms.values()) else 0
        similarity_factor = min(similarity / 1.5, 0.6)
        return float(round(term_factor + similarity_factor, 2))

    def _get_violation_details(self, category, detected_terms, similarity, threshold):
        """Coleta detalhes específicos da violação"""
        return {
            'termos_detectados': {
                'exatos': [
                    {
                        'termo_original': term['original_term'],
                        'variante_detectada': term['matched_variant']
                    } for term in detected_terms['regex_terms']
                ],
                'semanticos': [
                    {
                        'termo_referencia': term['term'],
                        'similaridade': term['similarity'],
                        'threshold': term['threshold']
                    } for term in detected_terms['semantic_terms']
                ]
            },
            'metricas_similaridade': {
                'valor': float(round(similarity, 2)),
                'threshold': float(threshold)
            }
        }

    def set_error(self, error_message):
        """Registra erros durante o processamento"""
        self.error = {
            'message': error_message,
            'compliance_status': None
        }
        self.compliance_status = False

    def to_dict(self):
        """Retorna resultados em formato dicionário serializável"""
        return {
            'texto_analisado': self.text,
            'versao_sistema': self.version,
            'conformidade_geral': self.compliance_status,
            'violacoes': self.violations if self.violations else None,
            'detalhes_analise': self.categories,
            'erro': self.error
        }


class ComplianceAnalyzer:
    """Sistema de análise de conformidade regulatória"""

    def __init__(self, embedding_method="weighted"):
        self.embedding_method = embedding_method
        self._carregar_configuracoes()
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.stemmer = RSLPStemmer()
        self._preprocessar_recursos()
        self.version = "3.2.1"

    def _carregar_configuracoes(self):
        """Carrega configurações das categorias de análise"""
        self.config = {
            "privacidade": {
                "texto_referencia": (
                    "violação de dados pessoais, coleta ilegal sem consentimento, "
                    "vazamento de informações sensíveis, descumprimento da LGPD"
                ),
                "threshold": 0.65,
                "threshold_semantico": 0.60,
                "termos_criticos": [
                    "sem consentimento", "vazamento de dados",
                    "acesso não autorizado", "monitoramento ilegal"
                ]
            },
            "etica": {
                "texto_referencia": (
                    "discriminação racial, discurso de ódio, incitação à violência, "
                    "preconceito religioso, xenofobia, assédio moral"
                ),
                "threshold": 0.60,
                "threshold_semantico": 0.55,
                "termos_criticos": [
                    "racismo", "ódio", "xenofobia", "machismo",
                    "discriminação", "discriminar"
                ]
            }
        }

    def _preprocessar_recursos(self):
        """Pré-processa termos críticos e embeddings"""
        # Stemming de termos
        for categoria in self.config.values():
            categoria['termos_stemmizados'] = [
                self._aplicar_stemming(_normalize_text(termo))
                for termo in categoria['termos_criticos']
            ]

        # Embeddings das categorias
        self.embeddings_categorias = {}
        for nome, config in self.config.items():
            embedding = self._gerar_embedding(config['texto_referencia'])
            self.embeddings_categorias[nome] = embedding / np.linalg.norm(embedding)

    def _aplicar_stemming(self, texto):
        """Aplica stemming com tratamento de exceções"""
        try:
            texto = re.sub(r'(\w+)(ções?|dores?|mentos?|vidades?)\b', r'\1', texto)
            return ' '.join([self.stemmer.stem(palavra) for palavra in texto.split()])
        except Exception as e:
            logger.error(f"Erro no stemming: {e}")
            return texto

    def _gerar_embedding(self, texto):
        """Gera embeddings para texto"""
        try:
            inputs = self.tokenizer(
                texto,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.embedding_method == "weighted":
                mask = inputs['attention_mask'].unsqueeze(-1)
                embedding = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            else:
                embedding = outputs.last_hidden_state[:, 0, :]

            return embedding.numpy().flatten()

        except Exception as e:
            logger.error(f"Erro na geração de embedding: {e}")
            return np.zeros(self.model.config.hidden_size)

    def analisar(self, texto):
        """Executa análise completa do texto"""
        resultado = ComplianceResult(texto, self.version)

        try:
            for categoria in self.config:
                embedding_texto = self._gerar_embedding(texto)
                embedding_norm = embedding_texto / np.linalg.norm(embedding_texto)

                similaridade = cosine_similarity(
                    [embedding_norm],
                    [self.embeddings_categorias[categoria]]
                )[0][0]

                termos_detectados = self._detectar_termos(texto, categoria)
                violacao = self._verificar_violacao(
                    similaridade,
                    termos_detectados,
                    self.config[categoria]['threshold']
                )

                resultado.add_category_result(
                    category=categoria,
                    similarity=similaridade,
                    threshold=self.config[categoria]['threshold'],
                    detected_terms=termos_detectados,
                    violation=violacao
                )

        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            resultado.set_error(str(e))

        return resultado

    def _detectar_termos(self, texto, categoria):
        """Detecta termos críticos com metadados completos"""
        texto_normalizado = _normalize_text(texto)
        texto_stemmizado = self._aplicar_stemming(texto_normalizado)

        # Detecção exata
        termos_exatos = []
        for original, stemmizado in zip(
                self.config[categoria]['termos_criticos'],
                self.config[categoria]['termos_stemmizados']
        ):
            if re.search(rf'\b{re.escape(stemmizado)}\b', texto_stemmizado):
                termos_exatos.append({
                    'original_term': original,
                    'matched_variant': stemmizado
                })

        # Detecção semântica
        termos_semanticos = []
        embedding_texto = self._gerar_embedding(texto)
        embedding_norm = embedding_texto / np.linalg.norm(embedding_texto)

        for termo in self.config[categoria]['termos_criticos']:
            embedding_termo = self._gerar_embedding(termo)
            embedding_termo_norm = embedding_termo / np.linalg.norm(embedding_termo)
            similaridade = cosine_similarity([embedding_norm], [embedding_termo_norm])[0][0]

            if similaridade > self.config[categoria]['threshold_semantico']:
                termos_semanticos.append({
                    'term': termo,
                    'similarity': float(round(similaridade, 2)),
                    'threshold': self.config[categoria]['threshold_semantico']
                })

        return {
            'regex_terms': termos_exatos,
            'semantic_terms': termos_semanticos
        }

    def _verificar_violacao(self, similaridade, termos_detectados, threshold):
        """Determina se ocorreu violação"""
        tem_termos = any(termos_detectados['regex_terms']) or any(termos_detectados['semantic_terms'])
        return (
                (similaridade >= threshold * 1.1) or
                (tem_termos and similaridade >= threshold * 0.8)
        )


def formatar_resultado(resultado):
    """Gera relatório formatado para exibição"""
    dados = resultado.to_dict()

    relatorio = []
    relatorio.append("\n=== ANÁLISE DE CONFORMIDADE ===")
    relatorio.append(f"Versão do sistema: {dados['versao_sistema']}")
    relatorio.append(f"Texto analisado: {dados['texto_analisado'][:150]}...")
    relatorio.append(f"\nStatus Geral: {'CONFORME ✅' if dados['conformidade_geral'] else 'NÃO CONFORME ❌'}")

    if dados['violacoes']:
        relatorio.append("\nVIOLAÇÕES DETECTADAS:")
        for categoria, detalhes in dados['violacoes'].items():
            relatorio.append(f"\n■ Categoria: {categoria.upper()}")
            relatorio.append(f"  - Nível de Risco: {detalhes['risk_score']:.2f}/1.0")
            relatorio.append(f"  - Motivos: {', '.join(detalhes['reasons'])}")

            if detalhes['details']['termos_detectados']['exatos']:
                relatorio.append("\n  TERMOS EXATOS ENCONTRADOS:")
                for termo in detalhes['details']['termos_detectados']['exatos']:
                    relatorio.append(f"    • Original: {termo['termo_original']}")
                    relatorio.append(f"      Detectado: {termo['variante_detectada']}")

            if detalhes['details']['termos_detectados']['semanticos']:
                relatorio.append("\n  TERMOS SEMÂNTICOS RELACIONADOS:")
                for termo in detalhes['details']['termos_detectados']['semanticos']:
                    relatorio.append(f"    • Termo de Referência: {termo['termo_referencia']}")
                    relatorio.append(
                        f"      Similaridade: {termo['similaridade']:.2f} (Threshold: {termo['threshold']:.2f})")

            relatorio.append("\n  MÉTRICAS DE SIMILARIDADE:")
            relatorio.append(f"    • Score: {detalhes['details']['metricas_similaridade']['valor']:.2f}")
            relatorio.append(f"    • Threshold: {detalhes['details']['metricas_similaridade']['threshold']:.2f}")

    relatorio.append("\n" + "=" * 60)
    return '\n'.join(relatorio)


if __name__ == "__main__":
    '''Exemplo de uso interno'''
    analisador = ComplianceAnalyzer()

    casos_teste = [
        "Monitoramento não declarado de áreas de descanso dos funcionários",
        "Política de contratação exclusiva para homens",
        "Armazenamento de dados biométricos sem autorização explícita"
    ]

    for texto in casos_teste:
        resultado = analisador.analisar(texto)
        print(formatar_resultado(resultado))