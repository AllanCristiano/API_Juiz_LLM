from fastapi import FastAPI
from pydantic import BaseModel
from relevancia import RelevanciaController
from ConstitucionalIA import ComplianceAnalyzer

app = FastAPI()

# Instancia dos controllers para cada funcionalidade
relevancia_controller = RelevanciaController()
compliance_analyzer = ComplianceAnalyzer()

# Modelos de dados para as requisições
class Consulta(BaseModel):
    pergunta: str
    resposta: str

class TextRequest(BaseModel):
    texto: str

# Rotas para a funcionalidade de relevância
@app.post("/avaliar")
def avaliar_consulta(consulta: Consulta):
    """
    Recebe "pergunta" e "resposta", processa a avaliação de relevância e retorna o relatório detalhado.
    """
    relevancia_controller.processar_consulta(consulta.pergunta, consulta.resposta)
    relatorio = relevancia_controller.exibir_relatorio(-1)
    return relatorio

@app.get("/historico")
def obter_historico():
    """Retorna o histórico completo de avaliações de relevância."""
    return {"historico": relevancia_controller.obter_historico()}

@app.get("/ajustar-limiar")
def ajustar_limiar():
    """Ajusta automaticamente o limiar com base no histórico de relevância e retorna o novo valor."""
    novo_limiar = relevancia_controller.ajustar_limiar()
    return {"novo_limiar": novo_limiar}

# Rotas para a funcionalidade de conformidade (Compliance)
@app.post("/analisar")
def analisar_texto(request: TextRequest):
    """
    Recebe um JSON com o campo "texto", executa a análise de conformidade e retorna o resultado.
    """
    resultado = compliance_analyzer.analisar(request.texto)
    return resultado.to_dict()

@app.get("/version")
def get_version():
    """Retorna a versão do sistema de análise de conformidade."""
    return {"versao": compliance_analyzer.version}
