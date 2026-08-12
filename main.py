r"""
Agente FastAPI que recebe uma pergunta em texto humano, usa LangChain (tools)
e LangGraph (grafo de roteamento) com o MetIQ para decidir qual tool acionar,
executa a tool e devolve a resposta.

Tools disponíveis:
  1) consultar_cotacao      -> API do cotador (OAuth2 client_credentials)
  2) buscar_imagem_cachorro -> https://dog.ceo/api/breeds/image/random
  3) consultar_sinistro     -> API de claims por CPF (Client Assertion / PFX)
  4) consultar_pagamentos   -> API de pagamentos por CNPJ do corretor,
                                paginada (Client Assertion / PFX)
  5) consultar_conhecimento -> RAG de dois MetIQ Q&A prontos (produtos ou
                                subscrição médica), MetIQ Client Certificate
  6) (nenhuma tool)         -> resposta geral direta do modelo

Como o grafo funciona (LangGraph):
  START -> agent -> [tem tool_call?] --sim--> tools -> [foi RAG?] --sim--> END
                              \--não------------------------\--não--> agent

  O MetIQ não suporta tool calling nativo (bind_tools). Por isso as tools são
  descritas no system prompt, e o modelo responde em JSON puro dizendo qual
  tool chamar. O nó "agent" faz esse parsing manual e monta um AIMessage com
  `.tool_calls` no formato padrão do LangChain — a partir daí o resto do
  grafo (tools_condition, ToolNode) funciona exatamente como antes.

  Exceção: quando a tool executada é consultar_conhecimento (RAG), o grafo
  pula direto pro fim em vez de voltar pro "agent" — a resposta do RAG já
  vem pronta em linguagem natural, e reformular de novo só arrisca perder
  nuance ou parafrasear errado. Ver _apos_tool().

Memória de curto prazo:
  O grafo é compilado com MemorySaver (checkpointer em memória, por processo).
  Cada thread_id mantém seu próprio histórico de mensagens, permitindo
  perguntas de acompanhamento ("quantas estão pendentes?") sem re-chamar a
  tool. É in-memory, então some se o processo reiniciar, e é isolada por
  thread_id — quem chama a API precisa reenviar o mesmo thread_id a cada
  pergunta da mesma conversa (a API gera um novo automaticamente se não vier
  nenhum, e devolve na resposta).
"""

import os
import re
import time
import json
import uuid
import base64
import hashlib
import logging
from typing import Optional, Literal

from dotenv import load_dotenv

load_dotenv()  # carrega variáveis do .env se ele existir (não faz nada em produção/AKS)

import jwt as pyjwt
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization.pkcs12 import load_key_and_certificates

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from metiq_utils import create_metiq_client
from langchain_metiq import MetIQ, MetIQConfig, MetIQCertificate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente")

# ---------------------------------------------------------------------------
# Configuração (tudo via variável de ambiente — nunca hardcode segredos aqui)
# ---------------------------------------------------------------------------
COTADOR_TOKEN_URL = os.getenv("COTADOR_TOKEN_URL", "")
COTADOR_CLIENT_ID = os.getenv("COTADOR_CLIENT_ID", "")
COTADOR_CLIENT_SECRET = os.getenv("COTADOR_CLIENT_SECRET", "")
COTADOR_SCOPE = os.getenv("COTADOR_SCOPE", "")
COTADOR_API_URL = os.getenv("COTADOR_API_URL", "")

# Headers específicos exigidos pelo APIM/gateway do cotador. Nunca hardcode
# a subscription key — ela vem sempre de variável de ambiente / Secret.
COTADOR_SUBSCRIPTION_KEY = os.getenv("COTADOR_SUBSCRIPTION_KEY", "")
COTADOR_USER_AGENT = os.getenv("COTADOR_USER_AGENT", "Mozilla/5.0")
COTADOR_ACCEPT = os.getenv(
    "COTADOR_ACCEPT", "application/json;odata.metadata=minimal;odata.streaming=true"
)
# Idem para o IP: é específico de ambiente/infra, então também vem por env var.
COTADOR_X_FORWARDED_FOR = os.getenv("COTADOR_X_FORWARDED_FOR", "")

# ---------------------------------------------------------------------------
# Config das APIs de sinistros (claims) e pagamentos — Brazil Data Hub System
# APIs. Autenticação via Client Assertion (JWT assinado com certificado PFX),
# compartilhada pelas duas.
# ---------------------------------------------------------------------------
DATAHUB_PFX_BASE64 = os.getenv("DATAHUB_PFX_BASE64", "")  # conteúdo do .pfx em base64
DATAHUB_PFX_PATH = os.getenv("DATAHUB_PFX_PATH", "")  # alternativa: caminho pro arquivo
DATAHUB_PFX_PASSWORD = os.getenv("DATAHUB_PFX_PASSWORD", "")
DATAHUB_CLIENT_ID = os.getenv("DATAHUB_CLIENT_ID", "")
DATAHUB_TENANT_ID = os.getenv("DATAHUB_TENANT_ID", "")
DATAHUB_SCOPE = os.getenv("DATAHUB_SCOPE", "")
DATAHUB_SUBSCRIPTION_KEY = os.getenv("DATAHUB_SUBSCRIPTION_KEY", "")
DATAHUB_USER_AGENT = os.getenv("DATAHUB_USER_AGENT", "Mozilla/5.0")
DATAHUB_X_FORWARDED_FOR = os.getenv("DATAHUB_X_FORWARDED_FOR", "")

CLAIMS_API_URL = os.getenv("CLAIMS_API_URL", "")
PAYMENTS_API_URL = os.getenv("PAYMENTS_API_URL", "")

CLAIMS_FIELDS = [
    "declaredCause",
    "claimedName",
    "claimedNationalId",
    "brokerName",
    "claimNumber",
    "occurrenceDate",
    "claimStatus",
    "insuredAmountValue",
    "certificateNumber",
]

PAYMENTS_FIELDS = [
    "insuredNationalId",
    "insuredName",
    "dueDate",
    "paymentStatusDescription",
    "externalPolicyId",
    "externalInstallmentId",
    "premiumAmount",
]

# ---------------------------------------------------------------------------
# Config das tools de Q&A (RAG) — cada domínio é um MetIQ potencialmente
# diferente, com seu próprio conjunto completo de parâmetros (use_case_id,
# client_id, endpoint, certificado, etc). Nada é compartilhado entre eles.
# ---------------------------------------------------------------------------
def _carregar_config_qa(prefixo: str) -> dict:
    return {
        "use_case_id": os.getenv(f"{prefixo}_USE_CASE_ID", ""),
        "client_id": os.getenv(f"{prefixo}_CLIENT_ID", ""),
        "endpoint": os.getenv(f"{prefixo}_ENDPOINT", ""),
        "subscription_key": os.getenv(f"{prefixo}_SUBSCRIPTION_KEY", ""),
        "cert_base64": os.getenv(f"{prefixo}_CERT_BASE64", ""),
        "cert_path": os.getenv(f"{prefixo}_CERT_PATH", ""),
        "cert_password": os.getenv(f"{prefixo}_CERT_PASSWORD", ""),
        "cert_scope": os.getenv(f"{prefixo}_CERT_SCOPE", ""),
        "tenant_id": os.getenv(f"{prefixo}_TENANT_ID", ""),
    }


QA_DOMINIOS = {
    "produtos": _carregar_config_qa("QA_PRODUTOS"),
    "subscricao_medica": _carregar_config_qa("QA_SUBSCRICAO_MEDICA"),
}

DOG_API_URL = "https://dog.ceo/api/breeds/image/random"

app = FastAPI(title="Agente MetLife - LangChain + LangGraph + MetIQ")



# ---------------------------------------------------------------------------
# Cache simples de token OAuth2 (evita pedir token novo a cada request)
# ---------------------------------------------------------------------------
_token_cache = {"access_token": None, "expires_at": 0}


def _obter_token_cotador() -> str:
    """Client Credentials Grant contra o Azure AD / Microsoft Identity.

    Se a API real usar client assertion via certificado (PFX) em vez de
    client_secret, troque o corpo do POST abaixo pela montagem do JWT
    assinado com o certificado.
    """
    if not COTADOR_TOKEN_URL:
        raise RuntimeError("COTADOR_TOKEN_URL não configurada")

    agora = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] > agora + 30:
        return _token_cache["access_token"]

    data = {
        "grant_type": "client_credentials",
        "client_id": COTADOR_CLIENT_ID,
        "client_secret": COTADOR_CLIENT_SECRET,
    }
    if COTADOR_SCOPE:
        data["scope"] = COTADOR_SCOPE

    with httpx.Client(timeout=10) as http:
        resp = http.post(COTADOR_TOKEN_URL, data=data)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"Endpoint de token OAuth2 não devolveu JSON válido. "
                f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
            )

    token = payload["access_token"]
    expires_in = payload.get("expires_in", 3600)
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = agora + int(expires_in)
    return token


# ---------------------------------------------------------------------------
# Autenticação via Client Assertion (certificado PFX) — usada pelas APIs de
# sinistros e pagamentos. O JWT (client_assertion) expira em 2 minutos, então
# é gerado toda vez que o token de acesso precisa ser renovado — nunca fica
# fixo. O access_token resultante, esse sim, fica em cache normalmente.
# ---------------------------------------------------------------------------
_datahub_token_cache = {"access_token": None, "expires_at": 0}
_datahub_pfx_cache = {"private_key": None, "cert": None}


def _carregar_pfx():
    if _datahub_pfx_cache["private_key"] is None:
        if DATAHUB_PFX_BASE64:
            pfx_bytes = base64.b64decode(DATAHUB_PFX_BASE64)
        elif DATAHUB_PFX_PATH:
            with open(DATAHUB_PFX_PATH, "rb") as f:
                pfx_bytes = f.read()
        else:
            raise RuntimeError(
                "Nenhum certificado configurado: defina DATAHUB_PFX_BASE64 ou DATAHUB_PFX_PATH"
            )

        senha = DATAHUB_PFX_PASSWORD.encode() if DATAHUB_PFX_PASSWORD else None
        private_key, certificate, _ = load_key_and_certificates(
            pfx_bytes, senha, backend=default_backend()
        )
        _datahub_pfx_cache["private_key"] = private_key
        _datahub_pfx_cache["cert"] = certificate

    return _datahub_pfx_cache["private_key"], _datahub_pfx_cache["cert"]


def _gerar_client_assertion() -> str:
    """Gera um JWT assinado (RS256) com o certificado PFX, válido por 2
    minutos, para autenticação via Client Assertion no Azure AD."""
    private_key, certificate = _carregar_pfx()

    cert_der = certificate.public_bytes(Encoding.DER)
    x5t = base64.urlsafe_b64encode(hashlib.sha1(cert_der).digest()).decode().rstrip("=")

    agora = int(time.time())
    aud = f"https://login.microsoftonline.com/{DATAHUB_TENANT_ID}/oauth2/token"

    claims = {
        "iss": DATAHUB_CLIENT_ID,
        "sub": DATAHUB_CLIENT_ID,
        "aud": aud,
        "jti": str(uuid.uuid4()),
        "nbf": agora,
        "iat": agora,
        "exp": agora + 120,
    }

    return pyjwt.encode(claims, private_key, algorithm="RS256", headers={"x5t": x5t})


def _obter_token_datahub() -> str:
    if not DATAHUB_TENANT_ID or not DATAHUB_CLIENT_ID:
        raise RuntimeError("DATAHUB_TENANT_ID / DATAHUB_CLIENT_ID não configurados")

    agora = time.time()
    if _datahub_token_cache["access_token"] and _datahub_token_cache["expires_at"] > agora + 30:
        return _datahub_token_cache["access_token"]

    assertion = _gerar_client_assertion()
    token_url = f"https://login.microsoftonline.com/{DATAHUB_TENANT_ID}/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": DATAHUB_CLIENT_ID,
        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "client_assertion": assertion,
    }
    if DATAHUB_SCOPE:
        # Endpoint v1 (/oauth2/token) espera "resource", não "scope" (que é
        # do endpoint v2.0). Derivamos o resource removendo o "/.default".
        data["resource"] = DATAHUB_SCOPE.replace("/.default", "").strip()

    with httpx.Client(timeout=10) as http:
        resp = http.post(token_url, data=data)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"{e}. Corpo da resposta (primeiros 500 chars): {resp.text[:500]!r}"
            ) from e
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"Endpoint de token (Client Assertion) não devolveu JSON válido. "
                f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
            )

    token = payload["access_token"]
    expires_in = payload.get("expires_in", 3600)
    _datahub_token_cache["access_token"] = token
    _datahub_token_cache["expires_at"] = agora + int(expires_in)
    return token


def _datahub_headers() -> dict:
    token = _obter_token_datahub()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": DATAHUB_USER_AGENT,
    }
    if DATAHUB_SUBSCRIPTION_KEY:
        headers["Ocp-Apim-Subscription-Key"] = DATAHUB_SUBSCRIPTION_KEY
    if DATAHUB_X_FORWARDED_FOR:
        headers["x-forwarded-for"] = DATAHUB_X_FORWARDED_FOR
    return headers


# ---------------------------------------------------------------------------
# Tools (LangChain) — o docstring de cada função vira a "description" que o
# modelo lê para decidir quando usar a tool. Escreva com cuidado.
# ---------------------------------------------------------------------------
@tool
def consultar_cotacao() -> dict:
    """Consulta o número de uma cotação no sistema do cotador. Use quando o
    usuário pedir uma cotação, número de cotação ou status de uma proposta."""
    if not COTADOR_API_URL:
        raise RuntimeError("COTADOR_API_URL não configurada")

    token = _obter_token_cotador()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": COTADOR_USER_AGENT,
        "Accept": COTADOR_ACCEPT,
    }
    if COTADOR_SUBSCRIPTION_KEY:
        headers["Ocp-Apim-Subscription-Key"] = COTADOR_SUBSCRIPTION_KEY
    if COTADOR_X_FORWARDED_FOR:
        headers["x-forwarded-for"] = COTADOR_X_FORWARDED_FOR

    body = {
        "productId": 168,
        "username": "11992014985",
        "brokerage": 30,
        "FirstPremiumBrokerage": 100,
        "items": [
            {
                "cnpj": "00.386.748/0001-74",
                "isPrincipal": True,
                "livesAmount": 200,
                "insuredCapital": 2000000.00,
                "groupProfile": 7,
                "coverages": [
                    {
                        "id": 862981,
                        "percentage": None,
                        "daily": None,
                        "capital": None,
                    }
                ],
                "assistances": [],
            }
        ],
        "brokers": [],
    }

    with httpx.Client(timeout=15) as http:
        resp = http.post(COTADOR_API_URL, headers=headers, json=body)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"API do cotador não devolveu JSON válido. "
                f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
            )

    item = payload.get("item", payload)
    numero_cotacao = item.get("quotationID") or item.get("quotationId") or item.get("id")

    return {
        "numero_cotacao": numero_cotacao,
        "data_cotacao": item.get("quotationDate"),
        "validade": item.get("validityDateQuotation"),
    }


@tool
def buscar_imagem_cachorro() -> dict:
    """Busca uma imagem aleatória de um cachorro. Use quando o usuário pedir
    uma foto ou imagem de cachorro/cão."""
    with httpx.Client(timeout=10) as http:
        resp = http.get(DOG_API_URL)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"API do dog.ceo não devolveu JSON válido. "
                f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
            )

    return {"url_imagem": payload.get("message")}


@tool
def consultar_sinistro(cpf: str) -> dict:
    """Consulta sinistros (claims) de uma pessoa segurada pelo CPF. Use
    quando o usuário pedir status, situação ou detalhes de um sinistro."""
    if not CLAIMS_API_URL:
        raise RuntimeError("CLAIMS_API_URL não configurada")

    cpf_digitos = re.sub(r"\D", "", cpf or "")
    headers = _datahub_headers()
    params = {"q": f"claimedNationalId=={cpf_digitos};limit==50;offset==0"}

    with httpx.Client(timeout=15) as http:
        resp = http.get(CLAIMS_API_URL, headers=headers, params=params)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"{e}. Corpo da resposta (primeiros 500 chars): {resp.text[:500]!r}"
            ) from e
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"API de sinistros não devolveu JSON válido. "
                f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
            )

    itens = payload.get("items", [])
    sinistros = [
        {campo: item.get("item", item).get(campo) for campo in CLAIMS_FIELDS}
        for item in itens
    ]
    return {"sinistros": sinistros}


@tool
def consultar_pagamentos(cnpj: str) -> dict:
    """Consulta parcelas/pagamentos de apólices pelo CNPJ do corretor. Use
    quando o usuário pedir status de pagamento, parcelas ou boletos de um
    corretor."""
    if not PAYMENTS_API_URL:
        raise RuntimeError("PAYMENTS_API_URL não configurada")

    cnpj_digitos = re.sub(r"\D", "", cnpj or "")
    headers = _datahub_headers()

    todos_itens = []
    offset = 0
    limite_pagina = int(os.getenv("PAYMENTS_PAGE_SIZE", "10"))
    max_paginas = 100  # trava de segurança contra loop infinito

    with httpx.Client(timeout=httpx.Timeout(10.0, read=60.0)) as http:
        for _ in range(max_paginas):
            body = {
                "brokerDocumentList": [cnpj_digitos],
                "limit": str(limite_pagina),
                "offset": str(offset),
            }
            resp = http.post(PAYMENTS_API_URL, headers=headers, json=body)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"{e}. Corpo da resposta (primeiros 500 chars): {resp.text[:500]!r}"
                ) from e
            try:
                payload = resp.json()
            except ValueError:
                raise RuntimeError(
                    f"API de pagamentos não devolveu JSON válido. "
                    f"Status: {resp.status_code}. Corpo (primeiros 300 chars): {resp.text[:300]!r}"
                )

            pagina = payload.get("items", [])
            todos_itens.extend(pagina)

            if len(pagina) < limite_pagina:
                break
            offset += limite_pagina

    pagamentos = [
        {campo: item.get("item", item).get(campo) for campo in PAYMENTS_FIELDS}
        for item in todos_itens
    ]
    return {"pagamentos": pagamentos}


_qa_clientes_cache = {}


def _materializar_certificado_qa(config: dict, dominio: str) -> str:
    """Garante um caminho de arquivo pro certificado desse domínio,
    decodificando o base64 se necessário (prioridade sobre cert_path)."""
    if config["cert_base64"]:
        caminho = f"/tmp/qa_cert_{dominio}.pfx"
        with open(caminho, "wb") as f:
            f.write(base64.b64decode(config["cert_base64"]))
        return caminho
    if config["cert_path"]:
        return config["cert_path"]
    raise RuntimeError(f"Nenhum certificado configurado pro domínio '{dominio}'")


def _obter_cliente_qa(dominio: str):
    """Reaproveita o cliente MetIQ entre chamadas — um por domínio, cada um
    com seu próprio conjunto completo de parâmetros."""
    if dominio not in _qa_clientes_cache:
        config = QA_DOMINIOS.get(dominio)
        if not config or not config["use_case_id"]:
            raise RuntimeError(f"Domínio '{dominio}' não configurado (use_case_id ausente)")

        cert_path = _materializar_certificado_qa(config, dominio)
        certificate_config = MetIQCertificate(
            certificate_path=cert_path,
            certificate_password=config["cert_password"],
            certificate_scope=config["cert_scope"],
            certificate_tenant=config["tenant_id"],
        )
        llm_config = MetIQConfig(
            use_case_id=config["use_case_id"],
            client_id=config["client_id"],
            endpoint=config["endpoint"],
            subscription_key=config["subscription_key"],
            certificate_config=certificate_config,
        )
        _qa_clientes_cache[dominio] = MetIQ(config=llm_config)

    return _qa_clientes_cache[dominio]


@tool
def consultar_conhecimento(dominio: Literal["produtos", "subscricao_medica"], pergunta: str) -> str:
    """Consulta uma base de conhecimento (RAG) especializada da MetLife pra
    dúvidas que não são sobre cotação, sinistro ou pagamento. Use domínio
    'produtos' para dúvidas sobre produtos MetLife: carências, coberturas,
    condições gerais, aceitação geral. Use domínio 'subscricao_medica' para
    dúvidas sobre subscrição médica: aceitação baseada em condições de saúde
    do segurado, doenças preexistentes, exames exigidos."""
    cliente = _obter_cliente_qa(dominio)
    try:
        resposta_rag = cliente.invoke(pergunta)
    except Exception as e:
        raise RuntimeError(f"Erro ao consultar RAG do domínio '{dominio}': {e}")

    # Devolve o texto do RAG tal como veio — sem passar de novo pelo LLM
    # principal pra "reformular" (o roteamento do grafo pula direto pro
    # fim depois dessa tool, ver _apos_tool).
    return resposta_rag


TOOLS = [
    consultar_cotacao,
    buscar_imagem_cachorro,
    consultar_sinistro,
    consultar_pagamentos,
    consultar_conhecimento,
]

# ---------------------------------------------------------------------------
# Modelo (MetIQ) — o MetIQ NÃO suporta bind_tools nativo (tool calling), ao
# contrário do ChatOpenAI. Por isso as tools são descritas em texto no
# system prompt, e o modelo é instruído a responder em JSON dizendo qual
# tool chamar. Esse padrão segue a documentação oficial "Building AI Agents
# using LangGraph & MetIQ" (Exercise 4: Tool Calling).
# ---------------------------------------------------------------------------
llm = create_metiq_client()
# llm = llm.bind_tools(TOOLS)  # MetIQ doesn't support native tool calling


def _build_system_prompt() -> SystemMessage:
    tools_desc = "\n".join(
        f"{i + 1}) {t.name}: {t.description}" for i, t in enumerate(TOOLS)
    )
    prompt = f"""Você é um assistente que responde perguntas de usuários.

Você pode usar as seguintes tools para ajudar a responder perguntas ou executar tarefas:
{tools_desc}

---
Formato de resposta:
{{
  "response_type": "<tool_call, answer ou error>",
  "tool_call": {{
    "name": "<nome da tool a usar>",
    "args": {{}}
  }},
  "message": "<sua resposta final ao usuário ou descrição do erro>"
}}
---

IMPORTANTE: Lembre-se do seguinte:
- Preferencialmente, use as tools para responder às perguntas.
- Caso nenhuma tool seja adequada para a pergunta, responda com base no seu conhecimento geral.
- Você DEVE responder SOMENTE em um dos formatos de resposta acima.
- NÃO use markdown ou blocos de código na resposta.
- Em caso de resposta JSON, NÃO use aspas simples, use aspas duplas.
"""
    return SystemMessage(content=prompt)


SYSTEM_PROMPT = _build_system_prompt()


def _try_parse_json(raw):
    """Tenta parsear JSON da resposta do MetIQ, lidando com cercas de código."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.replace("```json", "```").split("```")[1].strip() if "```" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def agent_node(state: MessagesState):
    messages = state["messages"]

    try:
        llm_response = llm.invoke(messages)  # MetIQ .invoke() devolve string, não AIMessage
    except Exception as e:
        logger.exception("Erro ao chamar MetIQ")
        llm_response = f"Error invoking LLM: {e}"

    ai_message = AIMessage(content="")
    parsed = _try_parse_json(llm_response)

    if parsed and parsed.get("response_type") == "tool_call":
        tool_call = parsed.get("tool_call", {})
        ai_message.tool_calls = [
            {
                "id": f"tool_call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
                "name": tool_call.get("name"),
                "args": tool_call.get("args", {}),
            }
        ]
        ai_message.content = llm_response
    elif parsed:
        # response_type "answer" ou "error": usa o texto final já extraído
        ai_message.content = parsed.get("message", llm_response)
    else:
        # Resposta não veio no formato JSON esperado; usa o texto bruto
        ai_message.content = llm_response

    return {"messages": [ai_message]}


# ---------------------------------------------------------------------------
# Grafo (LangGraph)
# ---------------------------------------------------------------------------
def _apos_tool(state: MessagesState):
    """Depois de uma tool rodar: normalmente volta pro 'agent' pra formular
    a resposta final em linguagem natural a partir dos dados brutos. Mas se
    foi a tool de RAG (consultar_conhecimento), a resposta já vem pronta —
    pula direto pro fim, sem passar de novo pelo LLM."""
    ultima = state["messages"][-1]
    if isinstance(ultima, ToolMessage) and ultima.name == "consultar_conhecimento":
        return END
    return "agent"


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")
# tools_condition olha a última mensagem: se tiver tool_calls, vai pra
# "tools"; se não tiver, vai pra END.
builder.add_conditional_edges("agent", tools_condition)
builder.add_conditional_edges("tools", _apos_tool)

memory_saver = MemorySaver()
grafo = builder.compile(checkpointer=memory_saver)


# ---------------------------------------------------------------------------
# Modelos de request/response
# ---------------------------------------------------------------------------
class PerguntaRequest(BaseModel):
    pergunta: str
    thread_id: Optional[str] = None


class RespostaAgente(BaseModel):
    resposta: str
    tool_usada: Optional[str] = None
    dados: Optional[str] = None
    thread_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=RespostaAgente)
def ask(req: PerguntaRequest):
    thread_id = req.thread_id or (time.strftime("%Y%m%d%H%M%S") + "-" + str(uuid.uuid4()))
    config = {"configurable": {"thread_id": thread_id}}

    # Verifica se essa thread já tem histórico — só nesse caso pulamos o
    # system prompt, pra ele não se repetir a cada pergunta da mesma sessão.
    try:
        estado_existente = grafo.get_state(config)
        mensagens_antes = estado_existente.values.get("messages", []) or []
    except Exception:
        mensagens_antes = []

    novas_mensagens = [HumanMessage(content=req.pergunta)]
    if not mensagens_antes:
        novas_mensagens = [SYSTEM_PROMPT] + novas_mensagens

    entrada = {"messages": novas_mensagens}

    try:
        resultado = grafo.invoke(entrada, config=config)
    except Exception as e:
        logger.exception("Erro ao executar o grafo")
        raise HTTPException(status_code=502, detail=f"Erro ao processar: {e}")

    mensagens = resultado["messages"]
    # Só olhamos as mensagens geradas NESTA chamada (a partir de onde a
    # thread estava antes) — evita reportar uma tool de um turno anterior
    # como se fosse desta pergunta.
    mensagens_novas = mensagens[len(mensagens_antes):]

    tool_usada = None
    dados = None
    for m in mensagens_novas:
        if isinstance(m, ToolMessage):
            tool_usada = m.name
            dados = m.content

    resposta_final = mensagens[-1].content

    return RespostaAgente(
        resposta=resposta_final,
        tool_usada=tool_usada,
        dados=dados,
        thread_id=thread_id,
    )
