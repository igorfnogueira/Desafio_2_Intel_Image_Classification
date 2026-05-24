# Intel Image Classification

Projeto de **Visão Computacional** para classificar imagens de cenas em **6 categorias** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`), usando o dataset [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

## O que o projeto faz

O notebook `intel_image_classification.ipynb` monta um fluxo completo em **PyTorch** (com suporte a GPU/CUDA):

1. Carrega e explora os dados (`seg_train` / `seg_test`)
2. Normaliza as imagens e aplica **data augmentation** só no treino
3. Treina três CNNs com complexidade crescente:
   - **Baseline** — rede simples de referência
   - **VGG-like** — mais profunda, com BatchNorm
   - **AdvancedIntelCNN** — arquitetura híbrida (stem + blocos VGG + residuais)
4. Compara os modelos, escolhe o melhor pela validação e avalia no teste (matriz de confusão e erros)

## Como baixar o dataset

Fonte oficial: **[Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)**

O arquivo compactado traz três pastas. Para este projeto, você precisa de **`seg_train`** (imagens com rótulo para treino) e **`seg_test`** (imagens com rótulo para teste). A pasta **`seg_pred`** contém imagens sem rótulo e não é usada no notebook atual.

### Opção 1 — Pelo site (mais simples)

1. Acesse o link do dataset e faça login na [Kaggle](https://www.kaggle.com/) (crie uma conta gratuita, se ainda não tiver).
2. Na página do dataset, clique em **Download** (pode pedir para aceitar as regras de uso na primeira vez).
3. Extraia o arquivo `.zip` baixado.
4. Copie as pastas **`seg_train`** e **`seg_test`** para a **raiz deste repositório** (mesmo nível do `intel_image_classification.ipynb`).

Dentro de cada pasta, as imagens ficam organizadas por classe:

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

### Opção 2 — Pela linha de comando (Kaggle API)

1. Instale a CLI: `pip install kaggle`
2. Gere um token em [Kaggle → Settings → API → Create New Token](https://www.kaggle.com/settings) e salve o arquivo `kaggle.json` em:
   - **Windows:** `C:\Users\<seu_usuario>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`
3. Na pasta do projeto, execute:

```powershell
kaggle datasets download -d puneet6060/intel-image-classification
```

4. Extraia o zip e mova `seg_train` e `seg_test` para a raiz do projeto.

> **Importante:** essas pastas **não** vão para o Git (estão no `.gitignore`) por causa do tamanho. Cada pessoa que clonar o repositório precisa baixar o dataset separadamente.

## Estrutura esperada do projeto

```
├── seg_train/          # baixado do Kaggle
├── seg_test/           # baixado do Kaggle
├── seg_pred/           # opcional (sem rótulo)
├── intel_image_classification.ipynb
├── requirements.txt
└── INSTALACAO.md       # ambiente virtual e PyTorch CUDA
```

## Como rodar

1. Baixe o dataset conforme a seção [Como baixar o dataset](#como-baixar-o-dataset).
2. Siga o [INSTALACAO.md](INSTALACAO.md) para criar o `.venv` e instalar as dependências.
3. Abra o notebook no Jupyter/VS Code, selecione o kernel do ambiente virtual e execute as células em ordem (**Run All** recomendado na primeira vez).

## Requisitos

- Python 3.11 ou 3.12
- GPU NVIDIA com CUDA (opcional, mas recomendado para treino)
