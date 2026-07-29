# Intel Image Classification

Projeto de **Visão Computacional** para classificar imagens de cenas em **6 categorias** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`), usando o dataset [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## O que o projeto faz

O notebook [`intel_image_classification.ipynb`](intel_image_classification.ipynb) monta um fluxo completo em **PyTorch** (com suporte a GPU/CUDA):

1. Carrega e explora os dados (`seg_train` / `seg_test`)
2. Normaliza as imagens e aplica **data augmentation** só no treino
3. Treina três CNNs com complexidade crescente:
   - **Baseline** — rede simples de referência
   - **VGG-like** — mais profunda, com BatchNorm
   - **AdvancedIntelCNN** — arquitetura híbrida (stem + blocos VGG + residuais)
4. Compara os modelos, escolhe o melhor pela validação e avalia no teste (matriz de confusão e erros)

## Resultados (execução salva no notebook)

Modelo selecionado pela validação: **AdvancedIntelCNN**.

| Modelo | Parâmetros | Acc. validação | Acc. teste |
|--------|------------|----------------|------------|
| **Advanced** | 14.360.390 | **87,42%** | **87,33%** |
| Baseline | 10.711.878 | 86,24% | 85,84% |
| VGG-like | 4.822.086 | 85,07% | 84,14% |

- Dataset rotulado: **17.034** imagens (14.034 treino + 3.000 teste), 6 classes
- Split interno do treino: 11.929 treino / 2.105 validação (`VAL_RATIO=0.15`, `SEED=42`)
- Melhor F1 por classe (teste): **forest 0,971** · menor: **glacier 0,828**
- Principais confusões: `buildings→street`, `glacier↔mountain`

### Onde ver o projeto funcionando

Abra o notebook com os **outputs já salvos** (gráficos e métricas embutidos):

| Conteúdo | Onde olhar no notebook |
|----------|-------------------------|
| Distribuição e amostras das classes | Fases 1–2 |
| Data augmentation | Fase 3 |
| Curvas de loss/acurácia por modelo | Fases 4–6 |
| Comparação dos 3 modelos | Fase 7 |
| Matriz de confusão e erros | Fase 8 |
| Síntese quantitativa | Fase 9 |

> Os pesos não são salvos em disco: para reproduzir o treino do zero, execute as células em ordem (GPU recomendada).

## Como baixar o dataset

Fonte oficial: **[Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)**

Você precisa de **`seg_train`** e **`seg_test`**. A pasta **`seg_pred`** (sem rótulo) é opcional e não é usada no notebook atual.

### Opção 1 — Pelo site

1. Acesse o link do dataset e faça login na [Kaggle](https://www.kaggle.com/).
2. Clique em **Download** e extraia o `.zip`.
3. Copie **`seg_train`** e **`seg_test`** para a raiz deste repositório (mesmo nível do notebook).

```
seg_train/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/

seg_test/
├── buildings/
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/
```

### Opção 2 — Kaggle API

1. `pip install kaggle`
2. Gere um token em [Kaggle → Settings → API](https://www.kaggle.com/settings) e salve `kaggle.json` em:
   - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
3. Na pasta do projeto:

```powershell
kaggle datasets download -d puneet6060/intel-image-classification
```

4. Extraia o zip e mova `seg_train` e `seg_test` para a raiz.

> **Importante:** dataset e `kaggle.json` **não** vão para o Git (`.gitignore`). Nunca versionar o token da Kaggle.

## Estrutura do repositório

```
├── intel_image_classification.ipynb   # pipeline + resultados salvos
├── requirements.txt
├── INSTALACAO.md                      # ambiente virtual e PyTorch/CUDA
├── README.md
├── seg_train/                         # local (ignorado pelo Git)
└── seg_test/                          # local (ignorado pelo Git)
```

## Como rodar

1. Baixe o dataset conforme a seção acima.
2. Siga o [INSTALACAO.md](INSTALACAO.md) para criar o `.venv` e instalar as dependências.
3. Abra `intel_image_classification.ipynb`, selecione o kernel do `.venv` e execute as células em ordem (**Run All** na primeira vez).

## Requisitos

- Python 3.11 ou 3.12
- GPU NVIDIA com CUDA (opcional, mas recomendado para treino)
