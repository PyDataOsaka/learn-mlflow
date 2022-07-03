---
marp: true
---
# MLflowによるハイパーパラメータ探索のロギング

---
## MLflowとは

* MLライフサイクル管理のためのOSS (DataBricks)
* 構成がシンプルで, 他のOSSとの組み合わせが容易（らしい）
* 以下の4つのコンポーネントを提供

  * MLflow Tracking - 実験管理
  * MLflow Projects - 再現可能なコード
  * MLflow Models - MLモデルの多様な環境へのデプロイ
  * MLflow Model Registry - MLモデルリポジトリ

* ここではMLflow Trackingを使用

---
## ハイパーパラメータ探索とMLflow

* MLflowは多数のハイパーパラメータの組み合わせに対する推論結果の管理を容易にする

---
## ノートブックの内容

* [Titanicデータセット](https://atmarkit.itmedia.co.jp/ait/articles/2007/02/news016.html)に対して[Catboost](https://catboost.ai/)による訓練と推論
* Scikit-learnの[パイプライン](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)を用いた処理の組み合わせ
* [Optuna](https://www.preferred.jp/ja/projects/optuna/)によるハイパーパラメータ探索
* MLflowによる実験管理
* MLflowによるハイパーパラメータ探索結果の管理

---
ここでは以下について説明します.

* Titanicデータセット
* Catboost
* Sklearnパイプライン
* Optuna
* MLflow Tracking

---
## Titanicデータセット

* タイタニック号沈没事故での搭乗者の生存・非生存の記録
* ノートブックでは[Seaborn](https://seaborn.pydata.org/#)のAPIでデータを取得
* Seabornのデータセットは元データを加工
  * Titanicデータセットについては[このスクリプト](https://github.com/mwaskom/seaborn-data/blob/master/process/titanic.py)で

---
![bg right:40%](https://64.media.tumblr.com/tumblr_m1eoahDgJ31qd3g3go1_1280.jpg)

* ノートブックで使用する特徴量

<style scoped>
table {
  font-size: 18px;
}
</style>

|列名|説明|種類|
|-|-|-|
|pclass|旅客クラス|順序|
|sex|性別|カテゴリ|
|age|年齢|連続値|
|sibsp|同乗している兄弟や配偶者の数|連続値|
|parch|同乗している親や子供の数|連続値|
|fare|旅客運賃|連続値|
|embark_town|出港地|カテゴリ|
|deck|[客室の階層](https://64.media.tumblr.com/tumblr_m1eoahDgJ31qd3g3go1_1280.jpg)|カテゴリ|
|survived|生存状況|カテゴリ, 目的変数|

---
## 前処理

---
## Catboost

---
## Catboostのハイパーパラメータ

---
## Optuna

---
## 探索空間の定義

---
## 目的関数の定義

---
## MLflow

---
## コンセプト
