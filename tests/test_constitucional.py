import os
import sys
import numpy as np
import pytest

# Adiciona o diretório raiz ao sys.path, se necessário
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ia_constitucional import (
    ComplianceAnalyzer,
    ComplianceResult,
    _normalize_text
)

# Teste para a função _normalize_text
def test_normalize_text():
    text = "Olá, Mundo! Çá?"
    normalized = _normalize_text(text)
    # Verifica se os acentos e cedilhas foram removidos
    assert "á" not in normalized
    assert "ç" not in normalized
    # Verifica se o texto está em minúsculas
    assert normalized == normalized.lower()

# Teste para o método set_error da classe ComplianceResult
def test_set_error():
    result = ComplianceResult("Texto de teste", "1.0")
    result.set_error("Erro de teste")
    d = result.to_dict()
    assert d["erro"] is not None
    assert d["conformidade_geral"] is False

# Classe dummy para substituir o método de geração de embeddings,
# evitando a dependência do modelo BERT durante os testes.
class DummyComplianceAnalyzer(ComplianceAnalyzer):
    def _gerar_embedding(self, texto):
        # Retorna um vetor fixo de dimensão 768
        return np.ones(768)

# Teste para o método analisar utilizando a classe dummy
def test_analisar_dummy():
    analyzer = DummyComplianceAnalyzer()
    text = "Teste de conformidade"
    result = analyzer.analisar(text)
    d = result.to_dict()
    # Verifica se os campos básicos estão presentes
    assert "texto_analisado" in d
    assert "versao_sistema" in d
    assert "conformidade_geral" in d
    assert "detalhes_analise" in d

# Teste para a função _detectar_termos com uma configuração controlada
def test_detect_terms(monkeypatch):
    analyzer = DummyComplianceAnalyzer()
    # Configuração mínima para testar a detecção de termos
    analyzer.config = {
        "privacidade": {
            "texto_referencia": "vazamento de dados",
            "threshold": 0.65,
            "threshold_semantico": 0.60,
            "termos_criticos": ["vazamento"]
        }
    }
    # Reprocessa os recursos para garantir que os termos stemmizados e embeddings de referência sejam gerados
    analyzer._preprocessar_recursos()
    # Testa a detecção com um texto que contém o termo crítico
    detected = analyzer._detectar_termos("Este sistema apresenta vazamento de informações", "privacidade")
    # Verifica se há termos exatos detectados
    assert len(detected["regex_terms"]) > 0
    # Verifica se a lista de termos semânticos é uma lista
    assert isinstance(detected["semantic_terms"], list)

if __name__ == "__main__":
    pytest.main()
