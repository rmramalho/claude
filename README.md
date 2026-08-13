<<<<<<< HEAD
# Agente com roteamento de tools (LangChain + LangGraph + MetIQ + FastAPI + AKS)

## O que o app faz

Expõe `POST /ask` recebendo `{"pergunta": "..."}`. Um grafo LangGraph decide,
via as tools definidas em LangChain, qual delas acionar:

```
START -> agent -> tem tool_call? --sim--> tools -> agent -> END
                        \--não-------------------------------> END
```

O nó `agent` chama o **MetIQ** (`metiq_utils.create_metiq_client()`). O MetIQ
não suporta tool calling nativo (`bind_tools`) — por isso as tools são
descritas em texto no system prompt, e o `agent` faz o parsing manual da
resposta JSON do modelo pra montar um `AIMessage.tool_calls` no formato
padrão do LangChain. A partir daí, `tools_condition` e `ToolNode` funcionam
normalmente: roteiam pro nó `tools`, executam a função Python real, e o
resultado volta pro `agent` até a resposta final.

- **consultar_cotacao** → chama a API do cotador (OAuth2 client_credentials)
  e retorna o número da cotação.
- **buscar_imagem_cachorro** → chama `https://dog.ceo/api/breeds/image/random`
  e retorna a URL da imagem.
- **consultar_sinistro** → consulta sinistros por CPF (Client Assertion/PFX).
- **consultar_pagamentos** → consulta parcelas por CNPJ do corretor, com
  paginação (Client Assertion/PFX).
- **consultar_conhecimento** → consulta um dos dois MetIQ Q&A prontos (RAG)
  — domínio `produtos` (carências/coberturas) ou `subscricao_medica`
  (aceitação por condição de saúde). Cada domínio tem seu próprio conjunto
  completo de parâmetros/certificado (podem ser MetIQs bem diferentes). A
  resposta do RAG é devolvida como veio, sem passar de novo pelo LLM
  principal pra reformular.
- Nenhuma tool → responde a dúvida geral diretamente com o modelo.

Também expõe `GET /health` para os probes do Kubernetes.

**Importante:** o `requirements.txt` agora instala pacotes via o índice do
JFrog Artifactory da MetLife (`langchain-metiq` só existe lá). Isso significa
que build e `pip install` só funcionam dentro da rede/VPN da MetLife, ou a
partir do Cloud Shell autenticado no tenant da empresa.

## 1. Rodar localmente (opcional, antes de containerizar)

```bash
pip install -r requirements.txt
cp .env.example .env   # preencha os valores reais
export $(cat .env | xargs)   # carrega as variáveis no shell
uvicorn main:app --reload
```

Teste:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "me manda uma foto de cachorro"}'
```

### Memória de curto prazo (`thread_id`)

A resposta traz um `thread_id`. Pra continuar a mesma conversa (ex: perguntar
"quantas estão pendentes?" depois de já ter pedido a lista de parcelas),
reenvie esse mesmo `thread_id` na próxima pergunta:

```json
{"pergunta": "quantas estão pendentes?", "thread_id": "20260725...-uuid-aqui"}
```

Se não enviar `thread_id`, cada pergunta é tratada como uma conversa nova
(sem memória do que foi perguntado antes). A memória é **in-memory** — dura
enquanto o processo estiver rodando, e é perdida se a aplicação reiniciar.

## 1b. Rodar como container local (VS Code + Docker Desktop)

Sem precisar de AKS nem ACR — só pra testar o container em si, na sua máquina.

Requer Docker Desktop instalado e rodando.

```bash
cp .env.example .env   # preencha os valores reais
docker compose up --build
```

Isso builda a imagem e sobe o container com as variáveis do `.env` já injetadas,
publicando em `http://localhost:8000`. Pra parar: `Ctrl+C`, ou `docker compose down`
numa outra aba do terminal.

Se preferir sem compose, os comandos equivalentes são:
```bash
docker build -t agente-app:local .
docker run --rm -p 8000:8000 --env-file .env agente-app:local
```

Teste igual ao passo anterior, trocando só a porta se necessário (é a mesma, 8000).

## 2. Build da imagem

Com Docker ou Podman (comandos idênticos, troque `docker` por `podman`):

```bash
docker build -t agente-app:v1 .
```

## 3. Push para o registry

```bash
docker tag agente-app:v1 <SEU_REGISTRY>/agente-app:v1
docker push <SEU_REGISTRY>/agente-app:v1
```

Troque `<SEU_REGISTRY>` pelo endereço real (ex: `meuacr.azurecr.io` ou
`metlife.jfrog.io/docker-local`).

## 4. Criar o Secret no cluster (nunca commitar isso em arquivo)

```bash
kubectl create secret generic agente-secrets \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=OPENAI_MODEL=gpt-4o-mini \
  --from-literal=COTADOR_TOKEN_URL=https://login.microsoftonline.com/<TENANT_ID>/oauth2/v2.0/token \
  --from-literal=COTADOR_CLIENT_ID=... \
  --from-literal=COTADOR_CLIENT_SECRET=... \
  --from-literal=COTADOR_SCOPE=api://<APP_ID>/.default \
  --from-literal=COTADOR_API_URL=https://sua-api-cotador.exemplo.com/cotacoes \
  --from-literal=COTADOR_SUBSCRIPTION_KEY=... \
  --from-literal=COTADOR_X_FORWARDED_FOR=... \
  --from-literal=DATAHUB_PFX_BASE64="$(base64 -w0 seu-certificado.pfx)" \
  --from-literal=DATAHUB_PFX_PASSWORD=... \
  --from-literal=DATAHUB_CLIENT_ID=... \
  --from-literal=DATAHUB_TENANT_ID=... \
  --from-literal=DATAHUB_SCOPE=... \
  --from-literal=DATAHUB_SUBSCRIPTION_KEY=... \
  --from-literal=CLAIMS_API_URL=... \
  --from-literal=PAYMENTS_API_URL=...
```

O `$(base64 -w0 seu-certificado.pfx)` já converte o arquivo `.pfx` pra base64 na
hora — não precisa fazer isso manualmente. **Nunca** commite o `.pfx` nem a
senha dele em nenhum arquivo do repositório; eles só devem existir como
Secret do Kubernetes ou no seu `.env` local (que já está fora do controle de
versão).

`COTADOR_USER_AGENT` e `COTADOR_ACCEPT` têm valor padrão no código (`Mozilla/5.0` e o
`Accept` do APIM) — só precisa passar no secret se quiser sobrescrever.

## 5. Ajustar e aplicar o k8s.yaml

Edite a linha `image:` em `k8s.yaml` com o mesmo caminho do passo 3. Se o
registry exigir autenticação de pull (ex: JFrog), descomente o bloco
`imagePullSecrets` e crie o secret correspondente:

```bash
kubectl create secret docker-registry registry-cred \
  --docker-server=<SEU_REGISTRY> \
  --docker-username=<USUARIO> \
  --docker-password=<TOKEN_OU_SENHA>
```

Aplique:

```bash
kubectl apply -f k8s.yaml
```

## 6. Verificar e testar

```bash
kubectl get pods
kubectl get service agente-app   # espere o EXTERNAL-IP
curl -X POST http://<EXTERNAL-IP>/ask \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "qual o número da minha cotação?"}'
```

## Notas

- Se a API do cotador usar client assertion via certificado (PFX) em vez de
  `client_secret`, ajuste a função `_obter_token_cotador()` em `main.py`.
- O campo lido da resposta do cotador (`numeroCotacao`) é um palpite razoável
  — ajuste conforme o contrato real da API.
- Nunca coloque segredos (`OPENAI_API_KEY`, client secret, etc.) direto no
  `k8s.yaml` ou em código — sempre via `Secret` do Kubernetes.

## Quickstart (Windows PowerShell)

Resumo rápido para desenvolver e testar localmente no Windows:

```powershell
# criar virtualenv e instalar dependências
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# copiar exemplo de env e editar
copy .env.example .env
# editar .env com suas credenciais (não commitar)

# rodar a aplicação
uvicorn main:app --reload

# testar endpoint
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"pergunta":"me manda uma foto de cachorro"}'
```

Se preferir usar Docker, veja as instruções acima na seção de container.
>>>>>>> claude/main
