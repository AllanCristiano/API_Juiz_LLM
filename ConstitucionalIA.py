import torch
import numpy as np
import re
import unicodedata
import logging
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
import nltk

# Configuração do sistema de logging
logging.basicConfig(
    filename='compliance_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Download de recursos necessários
nltk.download('rslp', quiet=True)


def _normalize_text(text):
    """Normalização de texto com preservação de contexto"""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = re.sub(r'(?<!\w)[^\w\s!?](?!\w)', '', text)
    return text.strip()


class ComplianceAnalyzer:
    """Sistema de Análise de Conformidade 3.0 com Detecção Hibrida"""

    def __init__(self, embedding_method="weighted"):
        self.embedding_method = embedding_method
        self._load_compliance_config()
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.stemmer = RSLPStemmer()
        self._precompute_critical_terms_data()
        self._precompute_embeddings()
        self.version = 3.0
        logger.info(f"Sistema inicializado. Versão: {self.version}")

    def _load_compliance_config(self):
        """Configurações atualizadas com novos parâmetros"""
        self.configs = {
            "privacy": {
                "embedding_text": (
                    "violação de dados pessoais, coleta ilegal sem consentimento, "
                    "vazamento de informações sensíveis, descumprimento da LGPD, "
                    "compartilhamento não autorizado, acesso indevido a registros, "
                    "monitoramento abusivo, retenção excessiva de informações, "
                    "segurança inadequada de dados, dados biométricos, registros médicos, "
                    "conversas privadas, vigilância oculta"
                ),
                "threshold": 0.65,
                "semantic_threshold": 0.60,
                "critical_terms": [
                    "sem consentimento", "vazamento de dados",
                    "acesso não autorizado", "monitoramento ilegal",
                    "violação de privacidade", "câmeras ocultas",
                    "monitoramento de funcionários", "compartilha informações sem autorização",
                    "dados biométricos", "registros médicos", "clandestino",
                    "conversas privadas", "vigilância oculta"  # Novos termos
                ]
            },
            "ethics": {
                "embedding_text": (
                    "discriminação racial, discurso de ódio, incitação à violência, "
                    "preconceito religioso, xenofobia, assédio moral, abuso verbal, "
                    "humilhação pública, comportamento antiético, preconceito de gênero, "
                    "intolerância, agressão verbal, ataques a minorias, desrespeito a direitos humanos, "
                    "viés de gênero, preferência sexual"  # Contexto adicional
                ),
                "threshold": 0.60,
                "semantic_threshold": 0.55,
                "critical_terms": [
                    "racismo", "ódio", "xenofobia", "machismo",
                    "discriminação", "discriminar", "discriminatório",
                    "intolerância", "intolerante", "violência verbal",
                    "ofensa racial", "comentário ofensivo", "preconceito",
                    "nao deveria ter direitos iguais", "discriminatórios",
                    "viés de gênero", "preferência sexual"  # Novos termos
                ]
            }
        }

    def _precompute_critical_terms_data(self):
        """Pré-processamento avançado de termos críticos"""
        for name, category in self.configs.items():
            # Stemming com tratamento especial
            category['stemmed_terms'] = [
                self._stem_text(_normalize_text(term))
                for term in category['critical_terms']
            ]

            # Embeddings para termos críticos
            category['term_embeddings'] = {
                term: self._get_embedding(term) / np.linalg.norm(self._get_embedding(term))
                for term in category['critical_terms']
            }
            logger.info(f"Pré-computados {len(category['critical_terms'])} termos para {name}")

    def _precompute_embeddings(self):
        """Pré-cálculo dos embeddings das categorias"""
        self.category_embeddings = {}
        for name, config in self.configs.items():
            embedding = self._get_embedding(config['embedding_text'])
            self.category_embeddings[name] = embedding / np.linalg.norm(embedding)
        logger.info("Embeddings das categorias pré-computados")

    def _stem_text(self, text):
        """Stemming aprimorado para termos complexos"""
        try:
            # Tratamento de sufixos específicos
            text = re.sub(r'(\w+)(ções?|dores?|mentos?|vidades?)\b', r'\1', text)
            return ' '.join([self.stemmer.stem(token) for token in text.split()])
        except Exception as e:
            logger.error(f"Erro no stemming: {e}", exc_info=True)
            return text

    def _get_embedding(self, text):
        """Geração de embeddings com contexto estendido"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.embedding_method == "cls":
                embedding = outputs.last_hidden_state[:, 0, :]
            elif self.embedding_method == "weighted":
                mask = inputs['attention_mask'].unsqueeze(-1)
                embedding = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            else:
                embedding = torch.mean(outputs.last_hidden_state, dim=1)

            return embedding.numpy().flatten()

        except Exception as e:
            logger.error(f"Erro no embedding: {e}", exc_info=True)
            return np.zeros(self.model.config.hidden_size)

    def _contains_critical_terms(self, text, category):
        """Detecção híbrida de termos críticos"""
        return (
                self._check_regex_terms(text, category) or
                self._check_semantic_terms(text, category)
        )

    def _check_regex_terms(self, text, category):
        """Busca por padrões textuais com stemming"""
        normalized = _normalize_text(text)
        stemmed = self._stem_text(normalized)

        for term in self.configs[category]['stemmed_terms']:
            if re.search(rf'\b{re.escape(term)}\b', stemmed):
                logger.info(f"Termo crítico detectado (regex): {term}")
                return True
        return False

    def _check_semantic_terms(self, text, category):
        """Detecção semântica contextual"""
        text_embed = self._get_embedding(text)
        text_embed /= np.linalg.norm(text_embed)

        for term, term_embed in self.configs[category]['term_embeddings'].items():
            similarity = cosine_similarity([text_embed], [term_embed])[0][0]
            if similarity > self.configs[category]['semantic_threshold']:
                logger.info(f"Similaridade semântica: {term} ({similarity:.2f})")
                return True
        return False

    def analyze(self, text):
        """Análise com lógica de decisão híbrida"""
        logger.info(f"Iniciando análise: {text[:50]}...")
        result = {'text': text, 'results': {}, 'version': self.version}

        try:
            for category in self.configs:
                embed = self._get_embedding(text)
                embed /= np.linalg.norm(embed)

                similarity = cosine_similarity([embed], [self.category_embeddings[category]])[0][0]
                has_terms = self._contains_critical_terms(text, category)

                # Lógica de decisão aprimorada
                violation = (
                        (similarity >= self.configs[category]['threshold'] * 1.1) or  # Similaridade alta
                        (has_terms and similarity >= self.configs[category]['threshold'] * 0.8)  # Combinação
                )

                result['results'][category] = {
                    'similarity': round(similarity, 2),
                    'threshold': self.configs[category]['threshold'],
                    'critical_terms': has_terms,
                    'violation': violation
                }

            result['compliance'] = not any(res['violation'] for res in result['results'].values())
            logger.info(f"Conformidade Geral: {result['compliance']}")

        except Exception as e:
            logger.error(f"Erro na análise: {e}", exc_info=True)
            result['error'] = str(e)

        return result


def print_analysis(result):
    """Saída formatada com destaque para violações"""
    print("\n" + "=" * 60)
    print(f"{' RELATÓRIO DE CONFORMIDADE ':=^60}")
    print(f"Versão: {result.get('version', 'N/A')}")
    print(f"Texto: {result['text'][:150]}{'...' if len(result['text']) > 150 else ''}")

    for cat, res in result.get('results', {}).items():
        status = "VIOLAÇÃO ❌" if res['violation'] else "Conforme ✅"
        print(f"\n{cat.upper():^60}")
        print(f"Similaridade: {res['similarity']:.2f} (Threshold: {res['threshold']:.2f})")
        print(f"Termos críticos: {'Detectados ✅' if res['critical_terms'] else 'Não detectados ✅'}")
        print(f"Status: {status}")

    print("\n" + "=" * 60)
    print(f"CONFORMIDADE GERAL: {'✅' if result.get('compliance', False) else '❌'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyzer = ComplianceAnalyzer(embedding_method="weighted")

    # Casos de teste atualizados
    test_cases = [
        "Monitoramento não declarado de áreas comuns",
        "Contratação com preferência por candidatos do sexo masculino",
        "Gravação de chamadas telefônicas sem autorização",
        "Compartilhamento de histórico médico com seguradoras",
        "Práticas salariais discriminatórias por etnia"
    ]

    for text in test_cases:
        result = analyzer.analyze(text)
        print_analysis(result)