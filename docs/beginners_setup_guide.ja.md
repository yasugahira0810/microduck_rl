# MicroDuck のシミュレーション環境を作った話（初心者向け記録）

> この文書は、Physical AI・MicroDuck・Hugging Face のすべてが初めての人が、
> 「ロボットの頭脳をクラウドで学習させる環境」を実際に構築した記録です。
> 2026-10-17 に MacBook Air (Apple M4) 上で行った作業を元にしています。

---

## 1. そもそも何をしようとしているのか

### MicroDuck とは

[MicroDuck](https://pollen-robotics.com/microduck/) は Pollen Robotics 社の
手のひらサイズの二足歩行ロボットです（身長約 25 cm、重さ約 800 g、$399）。
モーターは Dynamixel XL330 という小型サーボを 14 個搭載しています。

このロボットの面白いところは、**歩き方や起き上がり方を人間がプログラムするのではなく、
AI（強化学習）に学習させている**ことです。しかも学習は実機ではなく
**シミュレーション（物理エンジンの中の仮想ロボット）**で行い、
出来上がった「頭脳（ポリシー）」を実機に転送します。

### Physical AI と sim2real

- **Physical AI**: 画面の中ではなく物理世界で動く AI のこと。
- **強化学習（RL）**: 試行錯誤を何百万回も繰り返して「うまく動く方法」を
  自力で覚えさせる手法。転んだら減点、前に進めたら加点、の繰り返し。
- **sim2real（シム・トゥ・リアル）**: シミュレーションで学んだことを
  実機に持っていく技術。仮想世界で何度転んでもロボットは壊れないので、
  思い切り学習できるのが利点。仮想と現実のズレ（摩擦、モーターの癖など）を
  どう埋めるかが腕の見せどころで、このリポジトリはその「レシピ」が詰まっています。

### 登場する道具たち

| 名前 | 役割 | たとえるなら |
|---|---|---|
| **MuJoCo** | 物理エンジン。重力・摩擦・衝突を計算する | ロボットの「仮想世界」そのもの |
| **mjlab** | MuJoCo を GPU で数千個並列に回す学習基盤 | 仮想世界の大量複製装置 |
| **microduck_rl**（このリポジトリ） | MicroDuck 用の学習レシピ集 | 料理のレシピ本 |
| **PPO** | 強化学習アルゴリズムの名前 | 学習の「進め方」の作法 |
| **ポリシー（ONNX ファイル）** | 学習の成果物 = ロボットの頭脳 | 焼き上がったパン |
| **uv** | Python の環境構築ツール（超高速） | 調理器具のセットアッパー |
| **Hugging Face (HF)** | AI 界の GitHub。モデル共有＋クラウド実行基盤 | AI の倉庫兼レンタル厨房 |
| **HF Jobs** | HF 上で GPU 付きマシンを借りて処理を実行する仕組み | 時間貸しの厨房 |
| **wandb** | 学習の経過をグラフで記録・表示するサービス | 炊飯のタイマー兼温度記録計 |

---

## 2. なぜクラウド（HF Jobs）で学習するのか

強化学習では、仮想ロボットを**数千体同時に**動かして試行錯誤させます
（今回の設定では 4096 体）。これには NVIDIA の GPU（CUDA）が必須です。

ところが Mac（Apple Silicon）には CUDA が載っていません。
そこで公式が用意している `--hf-jobs` 機能を使い、
**コードの送信と指示だけ Mac から行い、実際の学習は Hugging Face の
クラウド GPU にやってもらう**構成にしました。

```
[あなたの Mac]  --ソースコードを送信-->  [Hugging Face の GPU マシン]
      ↑                                      ↓
      └────  学習済みポリシー・記録を受け取る  ┘
```

借りた GPU は **L4（$0.80/時間、1 分単位の課金）** です。

---

## 3. 実際にやったこと（時系列）

### 手順 0: 前提の確認

- マシン: MacBook Air (Apple M4 / メモリ 32 GB / macOS 26.2)
- ローカルでの GPU 学習は不可（CUDA 非搭載）→ HF Jobs 方針に決定

### 手順 1: uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Python 本体も依存パッケージも uv が面倒を見てくれるため、
システムの Python（3.9 と古い）を触らずに済みます。

### 手順 2: レシピ本（リポジトリ）のコピー

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl
```

### 手順 3: 依存関係のインストール

```bash
uv sync
```

PyTorch（AI フレームワーク）、warp-lang（GPU 物理演算）、wandb など
142 個の部品が自動で入ります。Mac でも問題なく解決できました。

### 手順 4: 動作確認（お金のかからない範囲で）

```bash
uv run list-envs   # 学習できるタスク一覧が出れば OK（46 種類あった）
```

歩行・起き上がり・前転・ボールキック・ローラースケート等の
タスクが登録されているのを確認。

### 手順 5: Hugging Face の準備

1. [huggingface.co](https://huggingface.co) でアカウント作成
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens) で
   **write 権限 + Jobs 権限**のトークン（合鍵）を作成
3. `uv run hf auth login` で Mac に保存

> ⚠️ **つまずきポイント 1**: 最初のトークンには Jobs 権限がなく、
> ジョブ送信時に 403 エラーになりました。
> トークンの編集画面で「Jobs」権限にチェックを入れて解決。
> 権限を変えてもトークン文字列は変わらないので再ログインは不要でした。

### 手順 6: 課金クレジットのチャージ

[Settings → Billing](https://huggingface.co/settings/billing) で $10 チャージ。

> ⚠️ **つまずきポイント 2**: トークンが正しくてもクレジット残高ゼロでは
> ジョブは送れません（"insufficient Jobs credits" エラー）。
> なお、失敗したジョブには課金されませんでした。

### 手順 7: wandb の準備

学習の記録係。無料アカウントを作り、API キーでログイン：

```bash
uv run wandb login    # キーはターミナルに直接貼る（チャット等に貼らない）
```

> ⚠️ **つまずきポイント 3**: wandb 未ログインのまま送ったジョブは
> 「API キーがありません」と言ってクラッシュしました。
> このリポジトリは学習記録に wandb を必須とする設計です。

### 手順 8: スモークテスト（本送信）

いきなり本番の学習（数時間）を回さず、まず **5 イテレーションだけ**
回して配管全体を検証します。お金も数十分の一ドル単位しかかかりません。

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 5 --hf-jobs --detach
```

結果: **成功**。仮想ロボット 4096 体での学習が 5 回転し、
成果物が Hugging Face 上に自動で保存されました。

- 学習の頭脳の素: `model_0.pt` / `model_4.pt`（チェックポイント）
- ロボット搭載形式: `exported/policy.onnx`
- 学習曲線: wandb の `mjlab_microduck` プロジェクトで閲覧可能

---

## 4. 送信ボタンの裏で何が起きているか

`--hf-jobs` を付けてコマンドを打つと、裏ではこんなことが起きています：

1. 手元のリポジトリをまるごと tar.gz に梱包（9.2 MB）
2. HF 上の自分専用の倉庫（非公開 dataset リポジトリ）にアップロード
3. HF のクラウドで GPU コンテナが起動
4. コンテナ内で `uv sync`（2 回目以降はキャッシュで高速）
5. 学習が走り、チェックポイントが 60 秒おきに HF に自動保存
6. 終了時に ONNX（ロボット搭載形式）までエクスポートしてくれる

学習が途中で死んでも課金はそこで止まり、チェックポイントは残る設計です。

---

## 5. 費用の実績と目安（L4 GPU = $0.80/時間の場合）

| 内容 | 時間の目安 | 費用の目安 |
|---|---|---|
| スモークテスト（今回やったもの） | 十数分 | 約 $0.1（十数円） |
| 歩行ポリシーが「使える」レベル | 1〜2 時間 | 約 $1〜2（150〜300 円） |
| 本格的な歩行・転倒リカバリ | 3〜6 時間 | 約 $3〜5 |
| トリック系 1 本（前転、キック等） | 〜1 時間 | $1 未満 |

安全装置として、ジョブには 12 時間のタイムアウトが付いています
（仮に暴走しても L4 なら最大約 $9.6 で自動停止）。

---

## 6. 次のステップ

### 本番の歩行学習を回す

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

学習の様子は wandb のグラフでリアルタイムに眺められます。
「平均報酬が上がっているか」「罰則項がすべて 0 以下か」が健康チェックの基本です。

### 学習済みポリシーを Mac 上で試す（無料・GPU 不要）

```bash
uv run scripts/infer_policy.py --walking <ポリシー.onnx>
```

CPU 版 MuJoCo で仮想ロボットをキーボード操作できます。
実機が届く前の「出荷前検査」に相当します。

### 実機が届いたら

エクスポートした ONNX を `robotctl policy add` で実機にインストールし、
自分が育てた歩き方をするロボットと対面できます。
コミュニティへの公開（`uv run publish`）も可能です。

---

## 7. セキュリティの注意（これをやった人へ）

- **API トークン・キーはパスワードと同じ**です。
  チャット・メール・GitHub には絶対に貼らないでください。
  今回の作業でも HF トークンを誤ってチャットに貼ったため、
  作業後に「失効 → 作り直し → 再ログイン」を推奨しています。
- トークンの失効: [Settings → Tokens](https://huggingface.co/settings/tokens) → Revoke
- 作り直したら `uv run hf auth login` で再ログインするだけで環境は復活します。

---

## 8. 困ったときのコマンド集

```bash
cd ~/Documents/microduck        # まずここに移動（このリポジトリ）

uv run list-envs                # タスク一覧（環境が生きているかの確認）
uv run hf auth whoami           # HF ログイン状態の確認

# 送信中・送信済みジョブの確認
uv run python -c "from huggingface_hub import HfApi; \
  print([j.id for j in HfApi().list_jobs()])"
```

- 課金の状況: https://huggingface.co/settings/billing
- ジョブの一覧・ログ: https://huggingface.co/jobs
- 学習曲線: https://wandb.ai （プロジェクト名 `mjlab_microduck`）
- 公式のレシピ解説（上級者向け）: このリポジトリの `README.md` と `AGENTS.md`
