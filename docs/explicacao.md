# Explicação do projeto (pt-BR)

Este documento explica, de forma didática, **o que cada parte do código faz** e **por que cada
decisão de projeto foi tomada**. O objetivo é que você consiga entender e defender o trabalho numa
conversa — inclusive justificar escolhas como "por que `ZZFeatureMap`", "por que `GroupKFold`",
"por que priorizar *recall*" e "o que é a TVD e por que ela importa".

> Terminologia: **golden** = circuito limpo (original); **infected** = circuito com trojan;
> **trojan** = porta(s) inserida(s) maliciosamente; **backdoor** = trojan que só age sob um
> **gatilho** (*trigger*) específico.

---

## 1. Visão geral e pergunta de pesquisa

O projeto é a ponte entre duas linhas:

1. **Detecção de trojans em circuitos quânticos** (análise de sensibilidade + classificação do QASM
   como texto), e
2. **Quantum Adversarial Machine Learning (QAML)** (impacto de ataques sobre a *acurácia de
   classificação* de um VQC e construção de um *backdoor* com gatilho).

**Modelo adversarial (software supply chain).** Supomos que uma dependência de compilação antes
confiável — por exemplo, um *passe de transpilação* do Qiskit — foi comprometida e passa a
**inserir operações durante a compilação** de um VQC já treinado. O atacante **não** controla o
código-fonte do algoritmo, nem o provedor de nuvem, nem o hardware. O **defensor** tem acesso às
versões *golden* (limpa) e *infected* e às suas **medições**. Para o trojan de porta única, esse
cenário está implementado literalmente: o ataque é um `TransformationPass` dentro de um
`PassManager` (ver §5). O *backdoor* com gatilho, por sua vez, é **montado diretamente** (sem
`PassManager`), modelando o circuito que um passe comprometido emitiria (ver §5).

As três perguntas:

- **(a)** Qual o impacto comportamental e sobre a **acurácia** de inserções em VQCs?
- **(b)** É possível um **gatilho** que inverta a classificação só para entradas específicas,
  preservando as demais (*backdoor*)?
- **(c)** Detectores **textuais simples** (BoW/TF-IDF + ML clássico) sobre o QASM distinguem VQCs
  limpos de infectados?

---

## 2. `data.py` — datasets, PCA e codificação

- **Datasets binários:** Iris (setosa × versicolor), dígitos 0 vs 1 (`load_digits`) e Breast
  Cancer. Todos reduzidos a um problema **binário** com rótulos em {0, 1}.
- **Por que PCA?** Cada *feature* é codificada em **um qubit** (codificação por ângulo). Como
  trabalhamos com 2 ou 4 qubits de dados (no máximo 5 contando a ancila do *backdoor*), reduzimos
  a dimensionalidade com `PCA(n_components = n_qubits)`.
- **Pipeline `StandardScaler → PCA → MinMaxScaler`**, ajustado **apenas no treino** (sem vazamento
  para o teste). O `MinMaxScaler` leva as *features* para a **faixa de codificação**.
- **Por que a faixa `[0, π/2]`?** Ambos os *feature maps* aplicam uma fase `P(2x)`, cujo período em
  `x` é `π`. Se usássemos `[0, π]`, os dois extremos da faixa codificariam **o mesmo estado**
  (ambiguidade), e no Iris binário as classes caem justamente nos extremos do 1º componente do PCA.
  Com `[0, π/2]` a codificação é **injetiva** (cada valor vira um estado distinto). Pontos de teste
  que caiam fora da faixa de treino são **recortados** (`clip`) para dentro do intervalo válido.

---

## 3. `vqc.py` — o classificador quântico variacional

### O modelo
`|ψ(x, θ)⟩ = A(θ) · U_φ(x) · |0…0⟩`, onde:

- `U_φ` é o **feature map** (`z_feature_map` ou `zz_feature_map`): codifica os dados em ângulos.
- `A(θ)` é o **ansatz** `RealAmplitudes` (camadas de `RY` + emaranhadores `CX`), com os parâmetros
  treináveis `θ`.
- A saída é a **expectativa da paridade** `⟨Z^⊗n⟩ = P(paridade par) − P(paridade ímpar)`, em
  `[-1, 1]`. A **classe prevista** é `1` se essa expectativa for positiva, senão `0`.

### Por que medir expectativa da paridade?
É exatamente a "medição de expectativa" pedida, e tem uma vantagem conceitual importante: como a
saída é **função da distribuição de bits medida**, o efeito de um trojan sobre a **distribuição**
(medido pela TVD, §6) e sobre o **rótulo** (acurácia) ficam diretamente comparáveis.

### Por que `ZFeatureMap` e `ZZFeatureMap`?
- `ZFeatureMap`: codificação **produto** (um qubit por *feature*, sem emaranhamento na codificação).
  Simples e rápida; é a base do *backdoor* exato (§5).
- `ZZFeatureMap`: adiciona termos de **emaranhamento** `ZZ` entre pares de qubits — codificação mais
  expressiva, frequentemente citada em QML. Comparar as duas mostra como o ataque depende da
  estrutura de codificação.

### Por que `EstimatorQNN` + `COBYLA`?
Medimos a **expectativa exata** de `Z^⊗n` com o estimador do Aer em modo *statevector*
(`default_precision = 0`), o que elimina ruído de amostragem **durante o treino** e o torna rápido
e determinístico. `COBYLA` é um otimizador sem gradiente, robusto para poucas variáveis — padrão em
VQCs pequenos. O ponto inicial é sorteado com semente fixa (reprodutibilidade).

### `compile_golden` — a "compilação limpa"
Depois de treinar, fixamos `θ`, adicionamos as medições e rodamos o **fluxo de compilação limpo**:
`transpile(..., basis_gates=BASIS_GATES, optimization_level=0)`. As *features* continuam
**simbólicas**. O resultado é o **artefato golden**. O infected será esse mesmo fluxo **mais** o
passe malicioso — é assim que o ataque de *supply chain* se manifesta na prática.

> Guardamos nos metadados quantas operações iniciais de cada qubit pertencem ao *feature map*, para
> rotular a posição de inserção como *encoding*, *interface* ou *ansatz*.

---

## 4. `io_qasm.py` — por que OpenQASM 3

Um VQC treinado ainda é um **programa paramétrico**: os pesos são números fixos, mas as *features*
são **entradas em tempo de execução**.

- **OpenQASM 2 não serve:** não representa parâmetros livres (o Qiskit levanta
  `QASM2ExportError: 'Cannot represent circuits with unbound parameters'`), e o `qelib1.inc` antigo
  não tem as portas `p`/`u` do nosso basis.
- **OpenQASM 3 serve:** declara as *features* como `input float[64] _x_i_;`. Cada arquivo é, então,
  **um classificador completo e implantável** — justamente o que um compilador comprometido emitiria.
  O *round-trip* (exportar → reparsear) é garantido por `qiskit-qasm3-import` e **testado**.

Também calculamos **estatísticas estruturais** (profundidade, tamanho, nº de qubits/bits, contagem
de portas) e um **SHA-256** por arquivo (inventário).

---

## 5. `trojan.py` — inserção de trojans e o backdoor

### Trojan de porta única (`SingleGateInsertionPass`)
Um `TransformationPass` **de verdade** insere **uma** porta de `{H, T, X, Y, Z}`. Três **modos**:

- **`random`**: qubit e posição aleatórios (semente controlada), uniformes entre as `n + 1`
  posições de um fio com `n` portas reais (antes de cada porta ou depois da última). Barreiras e
  medições não contam como posição.
- **`idle`**: um ponto onde, naquela "camada" da DAG, o qubit está **ocioso** enquanto outro
  qubit executa uma porta — inserção furtiva que não aumenta a profundidade naquele instante. Se o
  circuito não tem nenhum ponto ocioso (ex.: codificação `z` com 2 qubits), cai num ponto
  aleatório e o registro marca `placement = random-fallback`.
- **`controlled`**: posição configurável; o padrão é a **fronteira codificação/ansatz**.

> **Correção metodológica (modo `idle`).** A primeira versão contava como "ociosas" as camadas
> finais que só têm `barrier`/`measure` (nelas todo qubit parece ocioso) e contava várias vezes o
> mesmo ponto quando um qubit ficava ocioso por camadas seguidas. Resultado: **65%** das inserções
> `idle` caíam logo antes da medição, onde `Z`/`T` não fazem nada (TVD igual ao piso de
> amostragem). Isso deixava o modo `idle` artificialmente "fraco". Agora essas camadas são
> ignoradas e o sorteio é uniforme entre pontos de inserção **distintos**. Ainda sobram pontos no
> fim do fio, mas são legítimos: um qubit que terminou suas portas enquanto outros continuam está,
> de fato, ocioso. A coluna `at_wire_end` dos metadados registra se a inserção ficou depois da
> última porta do fio.

Cada variante gera **metadados estruturados**: porta, modo, qubit, posição (e se ficou no fim do
fio), profundidade e tamanho antes/depois (e seus deltas), família/origem, nº de qubits e bits
clássicos.

**Intuição física que os resultados confirmam:** portas **diagonais** na base Z (`Z`, `T`) comutam
com a medição de `Z`; se inseridas **logo antes da medição**, não mudam a classe. Mas inseridas na
**interface** (antes do ansatz) elas são "espalhadas" pelo ansatz e passam a afetar a saída. Já
`X`/`Y` invertem a paridade e têm grande impacto. Isso aparece nos gráficos `tvd_by_gate.png`, e os
números confirmam: `Z`/`T` logo antes da medição dão TVD 0,056 contra um piso de 0,055 e **nenhuma**
predição invertida; em qualquer posição anterior, TVD 0,29 e taxa de inversão 0,25.

### Backdoor com gatilho (`build_backdoor`) — *conditional trojan*
Segue a construção de **John et al. (2025)**. Diferente do trojan de porta única, ele **não** é um
`TransformationPass`: o circuito é montado diretamente (feature map → detector → ansatz → inversão
condicional) para modelar o que um passe comprometido emitiria, e não é desdobrado para o *basis*
limpo. O modelo de ameaça é o mesmo; só a implementação é mais direta. A ideia:

1. O **gatilho** é "as primeiras `k` *features* estão no topo da faixa de codificação (`π/2`)".
2. Logo **após o feature map e antes do ansatz**, um **detector constante** aplica, em cada qubit do
   gatilho, o **inverso da codificação avaliada no ângulo do gatilho** (`U_enc(π/2)†`). Uma entrada
   codificada exatamente em `π/2` volta a `|0⟩`.
3. Um `MCX` (controles-em-zero, via `X` em volta) **marca uma ancila**; a codificação é
   **recomputada**, deixando os qubits de dados intactos.
4. Depois do ansatz, a ancila é **medida no meio do circuito** e um `X` **classicamente controlado**
   inverte o qubit 0 — e, portanto, a **paridade / a classe** — **apenas** quando o gatilho disparou.

**Por que o detector tem que vir antes do ansatz?** Se viesse depois, o ansatz já teria "embaralhado"
a codificação, e o inverso `U_enc(π/2)†` não devolveria o qubit a `|0⟩`. (Esse foi, inclusive, um bug
que corrigimos durante o desenvolvimento.)

**Por que medição intermediária + controle clássico?** Uma versão puramente unitária (CX da ancila
no qubit 0 sem medir) **emaranha** a ancila com os dados; como não medimos a ancila, ela é "traçada
fora" e o qubit 0 vira um estado **misto**, o que reduz a magnitude da expectativa e vaza para
entradas limpas. Medir a ancila **colapsa** o gatilho para "disparou / não disparou", tornando a
inversão **nítida** — exatamente o espírito do *stealthy conditional trojan*.

**Exatidão e degradação honesta:** como `U_enc` é por qubit (produto) na codificação `Z`, o detector
é **exato** para `z` (ASR ≈ 1 com baixo vazamento em `k = 3`). Para a codificação **`zz`**
(emaranhada), o inverso por qubit é só **aproximado**, então o ataque **piora** — e nós **medimos e
documentamos** isso em vez de esconder. Por isso o experimento de backdoor roda sobre os modelos de
4 qubits com codificação `z`.

**O detector é um limiar suave, não um teste de igualdade.** "Volta a `|0⟩` em `π/2`" é verdade,
mas uma entrada *perto* de `π/2` também volta quase toda. Para a codificação `z` (2 repetições)
desfeita em `π/2`, um qubit que codifica `x` volta a `|0⟩` com probabilidade **exatamente
`sin²(x)`** (há teste no `pytest`):

| x | 0 | π/8 | π/4 | 3π/8 | π/2 |
|---|---|---|---|---|---|
| P(\|0⟩) | 0,00 | 0,15 | 0,50 | 0,85 | 1,00 |

A ancila dispara com o **produto** dessas probabilidades nos `k` qubits do gatilho, e a classe
inverte (por maioria dos *shots*) quando esse produto passa de 1/2. Com `k = 1`, portanto,
**qualquer entrada limpa com `x₀ > π/4` tem a classe invertida** — metade da faixa de codificação.
Isso explica a grande queda de acurácia limpa em `k = 1`.

**Seletividade vs. vazamento:** quanto **maior** `k`, menor o produto das probabilidades, então a
região que dispara encolhe para perto do canto "todos os `k` *features* em `π/2`" e o
**vazamento** sobre entradas limpas **cai** — a ASR permanece alta. Uma segunda fonte (menor) de
vazamento: quando a probabilidade de disparo fica entre 0 e 1, medir a ancila também colapsa
parcialmente os qubits de dados, mesmo quando o gatilho não dispara. É a história honesta do
gráfico `backdoor_asr_vs_clean.png`.

---

## 6. `sensitivity.py` — TVD, BC, BD e métricas de ML

Para um golden `G` e um infected `I` avaliados na **mesma** entrada `x`, comparamos as
**distribuições de bits medidas** `p = P_G(·|x)` e `q = P_I(·|x)`:

- **TVD** (*total variation distance*): `½ · Σ|p_i − q_i|`, em `[0, 1]`. **O que é e por que
  importa:** é a probabilidade máxima com que se consegue distinguir as duas distribuições por
  qualquer teste. TVD alta ⇒ o trojan mudou de fato o comportamento observável do circuito. É a
  métrica central da análise de sensibilidade.
- **BC** (*Bhattacharyya coefficient*): `Σ √(p_i q_i)`, em `[0, 1]` (1 = idênticas) — a
  "sobreposição" entre as distribuições.
- **BD** (*Bhattacharyya distance*): `−ln(BC)`, em `[0, ∞)`.

Tudo com **512 shots × 15 repetições**, reportando **média e desvio**. A TVD de cada modelo é
estimada nas **8 primeiras entradas de teste** (`n_eval_inputs: 8` no `configs/full.yaml`) para
caber no orçamento de simulação. É uma amostra pequena por modelo; a acurácia e a taxa de
inversão (abaixo) usam o conjunto de teste inteiro.

### Por que uma **linha de base golden × golden**?
Com número **finito** de *shots*, **duas rodadas do mesmo circuito** já diferem por **ruído de
amostragem**. Sem uma referência, um "TVD > 0" não significa nada. Por isso simulamos o golden
**duas vezes** (sementes independentes) e reportamos esse **piso**: só TVD **acima** dele indica
efeito real do trojan. (Isso vai além do artigo-base e é uma boa prática metodológica.)

### Métricas de QAML
- **Acurácia limpa** e **queda de acurácia** (golden − infected), por variante — via estimador
  **exato** (rápido e determinístico).
- **Taxa de inversão de predição** (`flip_rate`): fração das entradas de teste cujo rótulo previsto
  pelo infected difere do golden. **Por que, se já temos a queda de acurácia?** Porque a queda
  **subestima** o impacto em modelos fracos. Um `X` logo antes da medição inverte a paridade, então
  a acurácia `a` vira `1 − a`. Num modelo com `a = 0,57`, a "queda" é só 0,14, embora **todas**
  as predições tenham mudado. Na primeira rodada, 118 das 1.620 variantes tinham queda
  **negativa** (o trojan "melhorou" o modelo). A taxa de inversão mede o efeito do trojan sobre a
  saída, independentemente de o modelo ser bom.
- Para o backdoor: **ASR** (fração de entradas com gatilho cujo rótulo **inverte** em relação ao
  golden) e **acurácia limpa** do modelo com backdoor — via **amostragem** (o backdoor tem medição
  intermediária e porta classicamente controlada, que o estimador exato não cobre).

### Otimização de desempenho
A transpilação é o custo dominante. Então: (1) transpilamos **uma vez** por circuito e só
re-executamos com sementes diferentes; (2) **pré-simulamos o golden uma vez por modelo** e
reaproveitamos entre as 15 variantes daquele modelo. Sem isso, a sensibilidade levaria horas.

---

## 7. `detect.py` — detecção textual (a "Tabela 1")

Tratamos o **QASM como texto** (um documento) e classificamos **golden (0) vs infected (1)**.

- **Representações:** **BoW** (`CountVectorizer`) e **TF-IDF**, ambos com **unigramas e bigramas** e
  um *tokenizer* que preserva nomes de porta, registradores, números e colchetes (estrutura do QASM).
- **Classificadores:** `LogisticRegression`, `MultinomialNB`, `ComplementNB`, `LinearSVC`.

### Três esquemas de validação: `group`, `arch` e `stratified`
- **`group` — GroupKFold por modelo-base:** mantém **todas as variantes de um mesmo modelo
  treinado no mesmo fold**. Sem isso, variantes quase idênticas do mesmo modelo apareceriam no
  treino **e** no teste, inflando o desempenho (*vazamento por similaridade*).
- **`arch` — GroupKFold por arquitetura (*leave-architecture-out*):** mantém juntos **todos os
  modelos de uma mesma arquitetura** (codificação × repetições do ansatz × nº de qubits). Esse é o
  teste em **estruturas de circuito que o detector nunca viu** — a pergunta **realista** de
  implantação.
- **`stratified` — StratifiedKFold:** ignora as famílias; é a linha de base **otimista**.

> **Por que o `arch` foi adicionado.** O benchmark tem só **12 arquiteturas**, e cada uma aparece
> em **9 modelos** (3 datasets × 3 sementes). Modelos da mesma arquitetura geram QASM
> **estruturalmente idêntico**: só mudam os ângulos numéricos dos `ry`. Logo, o `group` sozinho
> ainda mostra ao detector, no treino, a mesma estrutura que ele vê no teste. Isso explica por que
> `group` e `stratified` davam praticamente o mesmo resultado, e por que a versão anterior deste
> texto exagerava ao dizer que o `group` testava "famílias que o detector nunca viu".

### Desbalanceamento (por que subamostrar **só no treino** + `class_weight`)
Há **muito mais infected do que golden** (1 golden : 15 infected). Para evitar um detector trivial
que rotula **tudo como infected**:

- **Subamostragem aleatória da classe majoritária só no treino de cada fold** (semente por fold),
  **preservando a proporção real no teste**. Nunca tocamos no conjunto de teste — isso é **testado**
  no `pytest`.
- Para os modelos **lineares**, usamos também `class_weight="balanced"`.

### Por que priorizar **recall** (mas reportar o conjunto todo)?
A classe positiva é **infected**. Um **falso negativo** (VQC infectado classificado como limpo) é o
erro **crítico** — um circuito comprometido passaria despercebido. Então **recall** é a métrica de
maior consequência. Mas **recall sozinho** é enganoso (o detector trivial "tudo infectado" tem
recall 1,0). Por isso reportamos **sempre em conjunto** acurácia **balanceada**, **F1**, **F1-macro**
e **ROC-AUC**, além das **matrizes de confusão**. A tabela final sai no formato **média ± desvio**.

### O que os resultados realmente mostraram (leitura honesta)
No *benchmark* completo, **BoW (`count`) + modelo linear** (LinearSVC/LogReg) detecta **quase
perfeitamente** — recall ≈ 0,96–1,0 e ROC-AUC ≈ 1,0 — sob **os três esquemas**, inclusive o
***leave-architecture-out*** (`arch`), e **mesmo no caso só com `H`** (`detection_hard_summary.csv`).
O motivo é instrutivo, e as duas razões valem para **todas** as arquiteturas (por isso sobrevivem ao
`arch`):

1. Os circuitos golden são compilados para um vocabulário pequeno (`ry, p, cx, h`), e `x/y/z/t` são
   tokens que um *compile* limpo **nunca** emite (fora do vocabulário).
2. **Todo golden tem exatamente tantos `h` quanto fases de codificação `p(2*_x_i_)`**: o feature map
   aplica `H` e depois `P(2x)` em cada qubit, a cada repetição (verificado nos 108 golden). Um `h`
   inserido quebra essa igualdade de contagem, e um modelo linear sobre contagens aprende isso de
   imediato. Por isso até o caso "difícil" só com `H` é fácil.

Já **TF-IDF e os Naive Bayes são bem mais fracos** (recall 0,18–0,44 sob `group`, 0,05–0,29 sob
`arch`), porque o TF-IDF **reduz o peso** justamente dos tokens estruturais frequentes que carregam o
sinal. Sob `arch`, o ROC-AUC deles (até 0,90) fica bem acima do recall: a **ordenação** sobrevive à
mudança de arquitetura, mas o **limiar de decisão** não se transfere.

Conclusão honesta: a detecção lexical é fácil **neste cenário muito regular**, não em geral. Um
atacante que ficasse dentro do vocabulário limpo **e** preservasse essas relações de contagem não é
coberto por este experimento. Em circuitos reais e diversos, um detector de "saco de tokens" seria
bem menos confiável, pois **não modela a semântica** do circuito.

---

## 8. `explain.py` + `_tree_shap.py` — SHAP

Ajustamos uma **floresta aleatória** que prevê um alvo (TVD média, queda de acurácia ou taxa de
inversão) a partir dos **atributos estruturais** da inserção (porta, modo, qubit, posição, se ficou
no fim do fio, deltas de profundidade/tamanho, família, codificação). O **SHAP** (valores de
Shapley, via *Tree SHAP*) atribui a cada previsão a contribuição de cada atributo; reportamos a
**importância global** (média do |SHAP|). Responde: **"o que mais afeta o impacto do trojan?"**

No resultado, o atributo mais importante é `gate_t`, e não `gate_x`/`gate_y`. Isso **não**
contradiz a física: a importância mede o quanto um atributo **separa** as previsões, e `T` é o caso
extremo (quase inerte), então "é `T` ou não" é a divisão mais informativa. `Y`, a posição e
`at_wire_end` vêm em seguida.

### Nota de portabilidade importante (decisão de projeto honesta)
Nesta máquina o **Windows Smart App Control** está **ligado** e **bloqueia DLLs nativas não
assinadas**. Isso impede importar `numba`/`llvmlite` (de que o pacote `shap` depende) **e** também
`sklearn.ensemble` (o `RandomForestRegressor` "oficial"). Para o projeto rodar mesmo assim, sem abrir
mão da metodologia:

1. Implementamos a **floresta aleatória** a partir de `sklearn.tree.DecisionTreeRegressor`
   (*bootstrap* + subamostragem de *features*) — `BootstrapForest`.
2. Implementamos o **Tree SHAP exato** (algoritmo *path-dependent* de **Lundberg et al., 2018** — o
   **mesmo** que a `TreeExplainer` do `shap` executa) em **NumPy puro**, sem dependência nativa.

O `explain.py` **usa a biblioteca `shap` quando ela está disponível** e cai para a nossa
implementação quando não está — e as duas produzem **os mesmos valores de Shapley**. Validamos a
corretude pela **propriedade de eficiência** (base + Σ SHAP = previsão do modelo), que bate até a
**precisão de máquina** (~1e-16) e tem **teste** no `pytest`. Como a versão em Python puro é mais
lenta, calculamos o SHAP sobre uma **subamostra** de linhas (a importância **global** é estável sob
subamostragem — prática padrão).

---

## 9. `plotting.py` — figuras

Paleta **categórica segura para daltônicos** (validada em OKLab), com rótulos/eixos sempre visíveis.
Geramos *violin + strip plots* (como no artigo), barras de importância SHAP, barras agrupadas de
métricas e as matrizes de confusão.

---

## 10. Reprodutibilidade

- **Sementes fixas** em tudo (NumPy, scikit-learn, Qiskit/Aer, transpiler).
- **Versões fixadas** no `requirements.txt`.
- Modo **`--quick`** valida todo o pipeline em minutos; modo **completo** roda o *benchmark* inteiro.
- Saídas em **CSV** (metadata, sensitivity, inventory, errors, detection, backdoor, SHAP) e
  **figuras PNG**.
- **Testes `pytest`:** formato de saída dos VQCs; *round-trip* de QASM; trojan altera a distribuição
  (TVD > baseline); o modo `idle` ignora camadas só de barreira/medição e sorteia pontos distintos;
  posições do modo `random` indexam só portas reais; o detector do *backdoor* segue `sin²(x)`;
  subamostragem não toca no teste; o esquema `arch` separa arquiteturas inteiras; propriedade de
  eficiência do Tree SHAP; e o pipeline completo em modo *quick* (marcado como `slow`).

---

## 11. Como defender as escolhas (resumo rápido)

| Pergunta | Resposta curta |
|---|---|
| Por que OpenQASM 3? | QASM 2 não representa parâmetros livres; o VQC treinado tem *features* como entradas. |
| Por que expectativa de paridade? | Liga diretamente efeito na **distribuição** (TVD) e no **rótulo** (acurácia). |
| Por que `ZZFeatureMap`? | Codificação emaranhada e expressiva, comum em QML; contrasta com a `Z` (produto). |
| Por que baseline golden×golden na TVD? | Sem ela, não dá para separar o trojan do **ruído de amostragem**. |
| Por que `GroupKFold`? | Evita vazamento por similaridade entre variantes do mesmo circuito-base. |
| Por que também o esquema `arch`? | Modelos da mesma arquitetura têm QASM estruturalmente idêntico; só deixando a arquitetura inteira de fora o detector é testado em estrutura nova. |
| Por que `flip_rate` além da queda de acurácia? | Um trojan que inverte a paridade leva `a` a `1 − a`: em modelos fracos a "queda" é pequena embora todas as predições mudem. |
| O gatilho do backdoor é exato? | Não: é um limiar suave (`sin²(x)` por qubit). Com `k = 1`, qualquer `x₀ > π/4` dispara; `k` maior torna o gatilho seletivo. |
| Por que priorizar recall? | Falso negativo (infectado visto como limpo) é o erro crítico — mas reportamos tudo para não premiar o detector trivial. |
| Por que o backdoor é condicional (mede ancila)? | A versão unitária vaza por emaranhamento; medir a ancila deixa o gatilho nítido. |
| Por que Tree SHAP próprio? | O Smart App Control bloqueia `numba`/`shap` e `sklearn.ensemble`; a implementação exata dá os mesmos valores e roda em qualquer máquina. |
