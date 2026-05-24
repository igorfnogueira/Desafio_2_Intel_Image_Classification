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

## Estrutura esperada

```
├── seg_train/          # treino (subpastas por classe)
├── seg_test/           # teste oficial
├── seg_pred/           # imagens sem rótulo (uso futuro)
├── intel_image_classification.ipynb
├── requirements.txt
└── INSTALACAO.md       # ambiente virtual e PyTorch CUDA
```

## Como rodar

1. Baixe o dataset do Kaggle e extraia `seg_train` e `seg_test` na raiz do projeto.
2. Siga o [INSTALACAO.md](INSTALACAO.md) para criar o `.venv` e instalar as dependências.
3. Abra o notebook no Jupyter/VS Code, selecione o kernel do ambiente virtual e execute as células em ordem (**Run All** recomendado na primeira vez).


## Requisitos

- Python 3.11 ou 3.12
- GPU NVIDIA com CUDA (opcional, mas recomendado para treino)
