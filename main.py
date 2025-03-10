from fastapi import FastAPI
from pydantic import BaseModel
from relevancia import RelevanciaController

app = FastAPI()
controller = RelevanciaController()

# Modelo de dados para a requisição: apenas "pergunta" e "resposta"
class Consulta(BaseModel):
    pergunta: str
    resposta: str

@app.post("/avaliar")
def avaliar_consulta(consulta: Consulta):
    """
    Endpoint que recebe a pergunta e a resposta, processa a consulta e retorna o resultado da avaliação em JSON.
    """
    # Processa a consulta usando o limiar padrão (definido internamente como 0.45)
    controller.processar_consulta(consulta.pergunta, consulta.resposta)
    # Obtém o relatório completo da última avaliação
    relatorio = controller.exibir_relatorio(-1)
    return relatorio

@app.get("/historico")
def obter_historico():
    """
    Endpoint que retorna o histórico de todas as avaliações realizadas.
    """
    return {"historico": controller.obter_historico()}

@app.get("/ajustar-limiar")
def ajustar_limiar():
    """
    Endpoint que ajusta automaticamente o limiar com base no histórico de escores e retorna o novo valor.
    """
    novo_limiar = controller.ajustar_limiar()
    return {"novo_limiar": novo_limiar}
