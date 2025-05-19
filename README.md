# 日本語の感情分析（WRIME）

## 1. 概要

日本語の感情分析を行う簡単なpythonファイルになります。

## 2. タスク
- テキストの感情極性を3クラスに分類（'positive', 'negative', 'neutral'）
- 使用したデータセット：WRIME [[link](https://huggingface.co/datasets/llm-book/wrime-sentiment)]
- 評価指標：QWK

## 3. 設定
- モデル：tohoku-nlp/bert-base-japanese-v3 [[link](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)]
- ランダムシード：42
- 訓練データのバッチサイズ：32
- 検証データのバッチサイズ：64
- 学習率：1e-5
- エポック数：5

## 4. 環境構築
以下のように、仮想環境を構築し、必要なパッケージをインストールしてください。
```bash
# uv環境の構築
uv init
uv python pin 3.11.4
uv venv
rm -r .git

# アクティベート
. .venv/bin/activate

# パッケージのインストール
uv add transformers==4.47.0 datasets==3.5.1 scikit-learn==1.0.2 torch==2.0.1+cu118
```

## 5. 実行
```bash
# GitHubからsentiment.pyをとってくる
git clone https://github.com/j329nish/Sentiment-Analysis.git

# 実行
python3 sentiment.py
```

## 6. 評価

| Epoch | Training Loss | Validation Loss | Qwk |
|-:|-:|-:|-:|
| 1 | 0.741700 | 0.624303 | 0.590028 |
| 2 | 0.578100 | 0.591946 | 0.615269 |
| 3 | 0.504400 | 0.598148 | 0.614122 |
| 4 | 0.447700 | 0.623986 | 0.615468 |
| 5 | 0.422200 | 0.624582 | 0.622660 |

(最終更新 2025/5/19)
