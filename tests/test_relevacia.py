import os
import sys
import numpy as np
import pytest

# Adiciona o diretório raiz ao sys.path, se necessário
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relevancia import (
    RelevanciaModel,
    AnalisadorRelevancia,
    auto_threshold,
    RelevanciaController
)

# Classe dummy para simular o comportamento do SentenceTransformer
class DummyTransformer:
    def encode(self, texts):
        """
        Se os textos são iguais, retorna o mesmo vetor; caso contrário, retorna vetores diferentes.
        Vetor fixo de dimensão 3 para simplificar os cálculos.
        """
        if texts[0] == texts[1]:
            v = np.array([1.0, 0.0, 0.0])
            return [v, v]
        else:
            v1 = np.array([1.0, 0.0, 0.0])
            v2 = np.array([0.0, 1.0, 0.0])
            return [v1, v2]

# Teste para a função preprocess_text
def test_preprocess_text():
    model = RelevanciaModel()
    input_text = "Olá, Mundo! Çá?"
    output_text = model.preprocess_text(input_text)
    # Verifica se acentos e caracteres especiais foram removidos
    assert "á" not in output_text
    assert "ç" not in output_text
    # O output esperado, de acordo com a lógica atual, é "ola mundo ca"
    assert output_text == "ola mundo ca"

# Teste para compute_similarity com vetores idênticos
def test_compute_similarity_identical():
    model = RelevanciaModel()
    vec = np.array([1.0, 0.0, 0.0])
    similarity = model.compute_similarity(vec, vec)
    assert np.isclose(similarity, 1.0)

# Teste para compute_similarity com vetores ortogonais
def test_compute_similarity_different():
    model = RelevanciaModel()
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    similarity = model.compute_similarity(vec1, vec2)
    assert np.isclose(similarity, 0.0)

# Teste para avaliar_relevancia quando pergunta e resposta são iguais
def test_avaliar_relevancia_identical():
    analisador = AnalisadorRelevancia()
    # Substitui o transformer real pelo dummy
    analisador.modelo = DummyTransformer()
    result = analisador.avaliar_relevancia("Teste", "Teste")
    # Como os textos são iguais, espera-se similaridade 1.0
    assert np.isclose(result["similaridade"], 1.0)
    assert result["relevante"] is True

# Teste para avaliar_relevancia quando pergunta e resposta são diferentes
def test_avaliar_relevancia_different():
    analisador = AnalisadorRelevancia()
    analisador.modelo = DummyTransformer()
    result = analisador.avaliar_relevancia("Teste", "Diferente")
    # Para textos diferentes, o dummy retorna vetores [1,0,0] e [0,1,0] → similaridade 0.0
    assert np.isclose(result["similaridade"], 0.0)
    assert result["relevante"] is False

# Teste para gerar_relatorio, garantindo que o dicionário contém as chaves esperadas
def test_gerar_relatorio():
    analisador = AnalisadorRelevancia()
    analisador.modelo = DummyTransformer()
    analisador.avaliar_relevancia("Teste", "Teste")
    relatorio = analisador.gerar_relatorio()
    expected_keys = ["pergunta", "resposta", "similaridade", "limiar", "relevante", "detalhes", "analise", "recomendacoes"]
    for key in expected_keys:
        assert key in relatorio

# Teste para ajustar_limiar_automaticamente
def test_ajustar_limiar():
    analisador = AnalisadorRelevancia()
    analisador.modelo = DummyTransformer()
    # Adiciona duas avaliações: uma com similaridade 1.0 (textos iguais) e outra com similaridade 0.0 (textos diferentes)
    analisador.avaliar_relevancia("Teste", "Teste")
    analisador.avaliar_relevancia("Teste", "Diferente")
    novo_limiar = analisador.ajustar_limiar_automaticamente()
    # O novo limiar deve ser um valor entre 0 e 1
    assert 0 <= novo_limiar <= 1

# Teste para o RelevanciaController
def test_controller():
    controller = RelevanciaController()
    controller.analisador.modelo = DummyTransformer()
    result = controller.processar_consulta("Teste", "Teste")
    relatorio = controller.exibir_relatorio()
    # Verifica se o relatorio possui a chave "pergunta" e se o resultado indica relevância
    assert "pergunta" in relatorio
    assert result["relevante"] is True

if __name__ == "__main__":
    pytest.main()
