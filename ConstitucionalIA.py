import torch
import numpy as np
import re
import unicodedata
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import RSLPStemmer
import nltk

# Baixar o pacote do RSLP se necessário
nltk.download('rslp')


def _normalize_text(text):
    """
    Normaliza o texto: converte para minúsculas, remove acentos e pontuação.
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    return text


class ComplianceAnalyzer:
    """Sistema especializado em análise de privacidade e ética"""

    def __init__(self, embedding_method="weighted"):
        """
        Parâmetro embedding_method:
            - "cls": utiliza o token [CLS] como embedding
            - "weighted": utiliza a média ponderada considerando a attention mask
            - "mean": utiliza a média simples dos tokens
        """
        self.embedding_method = embedding_method
        self._load_compliance_config()
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self._precompute_embeddings()
        # Inicializa o stemmer para português
        self.stemmer = RSLPStemmer()
        # Pré-computa os termos críticos stemmizados para cada categoria
        self._precompute_stemmed_critical_terms()
        self.version = 1.3

    def _load_compliance_config(self):
        """Configuração detalhada dos princípios de privacidade e ética"""
        self.configs = {
            "privacy": {
                "embedding_text": (
                    "violação de dados pessoais, coleta ilegal sem consentimento, "
                    "vazamento de informações sensíveis, descumprimento da LGPD, "
                    "compartilhamento não autorizado, acesso indevido a registros, "
                    "falta de transparência no uso de dados, monitoramento abusivo, "
                    "retenção excessiva de informações, segurança inadequada de dados"
                ),
                "threshold": 0.76,  # Valor inicial, podendo ser calibrado
                "critical_terms": [
                    "sem consentimento", "vazamento de dados",
                    "acesso não autorizado", "monitoramento ilegal",
                    "violação de privacidade", "câmeras ocultas", "monitoramento de funcionários",
                    "compartilha informações sem autorização"
                ]
            },
            "ethics": {
                "embedding_text": (
                    "discriminação racial, discurso de ódio, incitação à violência, "
                    "preconceito religioso, xenofobia, assédio moral, abuso verbal, "
                    "humilhação pública, comportamento antiético, preconceito de gênero, "
                    "intolerância, agressão verbal, ataques a minorias, desrespeito a direitos humanos"
                ),
                "threshold": 0.65,
                "critical_terms": [
                    "racismo", "ódio", "xenofobia", "machismo",
                    "discriminação", "discriminar", "discriminatório",
                    "intolerância", "intolerante", "violência verbal",
                    "ofensa racial", "comentário ofensivo", "preconceito",
                    "nao deveria ter direitos iguais"
                ]
            }
        }

    def _precompute_stemmed_critical_terms(self):
        """
        Pré-processa os termos críticos para cada categoria aplicando normalização e stemming.
        """
        for category, config in self.configs.items():
            stemmed_terms = [self._stem_text(_normalize_text(term)) for term in config["critical_terms"]]
            self.configs[category]["stemmed_critical_terms"] = stemmed_terms

    def _stem_text(self, text):
        """
        Aplica o stemming ao texto: divide em tokens, aplica o stemmer e junta novamente.
        """
        tokens = text.split()
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def _precompute_embeddings(self):
        """Pré-cálculo dos embeddings para todas as categorias"""
        self.embeddings = {}
        for category, config in self.configs.items():
            embedding = self._get_embedding(config["embedding_text"])
            self.embeddings[category] = embedding / np.linalg.norm(embedding)

    def _get_embedding(self, text):
        """
        Gera embedding vetorial para o texto utilizando a estratégia definida.
        """
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, max_length=256, padding='max_length'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.embedding_method == "cls":
            embedding = outputs.last_hidden_state[:, 0, :]
        elif self.embedding_method == "weighted":
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        else:
            embedding = torch.mean(outputs.last_hidden_state, dim=1)

        return embedding.numpy().flatten()

    def _contains_critical_terms(self, text, category):
        """
        Verifica a presença de termos críticos no texto utilizando normalização, stemming e expressões regulares.
        """
        normalized_text = _normalize_text(text)
        stemmed_text = self._stem_text(normalized_text)
        for stemmed_term in self.configs[category].get("stemmed_critical_terms", []):
            pattern = r'\b' + re.escape(stemmed_term) + r'\b'
            if re.search(pattern, stemmed_text):
                return True
        return False

    def calibrate_threshold(self, texts, category, factor=1.0):
        """
        Calcula um threshold dinâmico com base em um conjunto de textos de calibração.
        Utiliza a fórmula: threshold = média + (factor * desvio padrão)
        """
        similarities = []
        for text in texts:
            text_embedding = self._get_embedding(text)
            text_embedding /= np.linalg.norm(text_embedding)
            similarity = cosine_similarity([text_embedding], [self.embeddings[category]])[0][0]
            similarities.append(similarity)
        mean_val = np.mean(similarities)
        std_val = np.std(similarities)
        new_threshold = mean_val + factor * std_val
        print(
            f"Calibração para {category}: média = {mean_val:.2f}, std = {std_val:.2f}, novo threshold = {new_threshold:.2f}")
        self.configs[category]["threshold"] = new_threshold

    def analyze(self, text):
        """Executa análise completa de conformidade"""
        results = {}

        for category, config in self.configs.items():
            text_embedding = self._get_embedding(text)
            text_embedding /= np.linalg.norm(text_embedding)
            similarity = cosine_similarity([text_embedding], [self.embeddings[category]])[0][0]
            has_critical_terms = self._contains_critical_terms(text, category)

            # Se houver alta similaridade ou termos críticos, há violação
            violation = similarity >= config["threshold"] or has_critical_terms

            results[category] = {
                "similarity": similarity,
                "threshold": config["threshold"],
                "critical_terms_found": has_critical_terms,
                "violation": violation
            }

        # Conformidade geral só é True se não houver nenhuma violação
        is_compliant = not any(res["violation"] for res in results.values())

        return {
            "text": text,
            "results": results,
            "compliance": is_compliant,
            "version": self.version
        }


def print_analysis(result):
    """Exibe os resultados formatados"""
    compliance_str = "SIM ✅" if result['compliance'] else "NÃO ❌"
    print("\n" + "=" * 60)
    print(f"{'ANÁLISE DE CONFORMIDADE':^60}")
    print("=" * 60)
    print(f"Versão do Sistema: {result['version']:.1f}")
    print("-" * 60)
    print("Texto analisado:")
    print(result['text'])
    print("-" * 60)

    for category, res in result['results'].items():
        category_title = "PRIVACIDADE" if category == "privacy" else "ÉTICA"
        critical_str = "Detectado ✅" if res['critical_terms_found'] else "Não detectado ❌"
        violation_str = "VIOLAÇÃO ❌" if res['violation'] else "Sem violação ✅"

        print(f"{category_title}:")
        print(f"  - Similaridade: {res['similarity']:.2f} (Limite: {res['threshold']:.2f})")
        print(f"  - Termos Críticos: {critical_str}")
        print(f"  - Status: {violation_str}")
        print("-" * 60)

    print(f"Conformidade Geral: {compliance_str}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyzer = ComplianceAnalyzer(embedding_method="weighted")

    # Conjuntos de calibração atualizados para melhor ajuste dos thresholds
    calib_privacy = [
        "Coletar dados pessoais sem consentimento é inaceitável",
        "Compartilhamento não autorizado de informações é um problema",
        "Vazamento de dados coloca em risco a privacidade",
        "A empresa compartilha informações sem autorização"
    ]

    calib_ethics = [
        "Discriminação racial é condenável",
        "Comentários de ódio não devem ser tolerados",
        "Discriminar com base na nacionalidade é inaceitável",
        "Este grupo de pessoas não deveria ter direitos iguais"
    ]

    analyzer.calibrate_threshold(calib_privacy, "privacy", factor=1.0)
    analyzer.calibrate_threshold(calib_ethics, "ethics", factor=1.0)

    # Exemplos de teste
    test_cases = [
        "Coletar dados pessoais sem consentimento é aceitável",
        "A empresa compartilha informações sem autorização",
        "Monitoramento de funcionários através de câmeras ocultas",
        "Este grupo de pessoas não deveria ter direitos iguais",
        "mulheres sao inferiores aos homens", "Não a pessoas inferiores"
    ]

    for text in test_cases:
        analysis = analyzer.analyze(text)
        print_analysis(analysis)
