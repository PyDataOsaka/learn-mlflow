---
marp: true
---
# MLflowによるハイパーパラメータ探索のロギング

---
## 今回の発表について

* 目標
* 構成要素の説明
  * Titanicデータセット
  * データ前処理とSklearnパイプライン
  * Catboost
  * Optuna
  * MLflow Tracking
* デモ用に作成したJupyter notebookの実演

---
## 目標

* ハイパーパラメータ探索結果の可視化
  * MLflow Trackingによる実験管理の基礎を学ぶ

## ノートブックの内容

* [Titanicデータセット](https://atmarkit.itmedia.co.jp/ait/articles/2007/02/news016.html)に対する[Catboost](https://catboost.ai/)による訓練と推論 ([catencoder.ipynb](./catencoder.ipynb))
  * Scikit-learnの[パイプライン](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)を用いた処理の組み合わせ
* [Optuna](https://www.preferred.jp/ja/projects/optuna/)によるハイパーパラメータ探索 ([optuna.ipynb](./optuna.ipynb))
* MLflowによる実験管理 ([mlflow-tracking.ipynb](./mlflow-tracking.ipynb))
* MLflowによるハイパーパラメータ探索結果の管理 ([mlflow-optuna.ipynb](./mlflow-optuna.ipynb))

---
## 構成要素の説明

* Titanicデータセット
* 前処理
* Sklearnパイプライン
* Catboost
* Optuna
* MLflow Tracking

---
## Titanicデータセット

* タイタニック号沈没事故での搭乗者の属性の記録
  * しばしばデータ分析や機械学習のチュートリアルで使用される
  * ここでは搭乗者の生存・非生存を他の属性から予測
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

* 使用する属性とクラスラベルの抽出

```python
df = sns.load_dataset('titanic') # データセットを取得
feature_names = [  # 使用する属性のリスト
    'class',
    'sex',
    'age',
    'sibsp',
    'parch',
    'fare',
    'embark_town',
    'deck',
]
df_x = df[feature_names]  # 属性
df_y = df['survived']     # 生存・非生存
```
---

* カテゴリ値を数値に変換するエンコーダを定義
  * Catboostが自動的にやってくれるかもしれないが念のため

```python
mapping = [
    {"col": "class", "mapping": {"First": 0, "Second": 1, "Third": 2}},
    {"col": "sex", "mapping": {"male": 0, "female": 1}},
    {"col": "embark_town", "mapping": {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}},
    {"col": "deck", "mapping": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}},
]

encoder = ce.OrdinalEncoder(cols=cols, mapping=mapping, handle_unknown='value')
df_x_enc = encoder.fit_transform(df_x)
```

* Catboostのカテゴリ値は整数値とする必要があるが, `OrdinalEncoder`はデフォルトで`float64`を出力
* `OrdinalEncoder`のコンストラクタに`dtype=int`とすれば良いが,
  ここではより汎用的な方法であるSklearn pipelineを用いる

---
## [Sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

* 様々な前処理と機械学習モデルを組み合わせるためのクラス
* 組み合わせたパイプラインは一つの機械学習モデルのように扱うことができる
* 例：入力スケーリング＋SVM

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(X_train, y_train)
```

* 独自の処理を実装可能

---
* 前処理をSklearn piplineに挿入可能にする
  * `transform()`と`fit_transform()`を持つクラスを定義

```python
class IntOrdEncoder(ce.OrdinalEncoder):
    def __init__(self, cols, mapping, handle_unknown):
        super().__init__(cols=cols, mapping=mapping, handle_unknown=handle_unknown)
        self.cols = cols

    def transform(self, *args, **kwargs):
        # 省略, fit_transformとほとんど同じ

    def fit_transform(self, *args, **kwargs):
        x = super().fit_transform(*args, **kwargs)
        for col in self.cols:
            # floatをintに変換しないとCatboostがエラーを出す
            x[col] = x[col].astype(int)

        return x
```

---
## [Catboost](https://catboost.ai)

* Yandexが開発した勾配ブースティングのPythonライブラリ（[arxiv](https://arxiv.org/abs/1706.09516)）
* 特徴は以下の通り

1. Great quality without parameter tuning
2. Categorical features support
3. Fast and scalable GPU version
4. Improved accuracy
5. Fast prediction

---

```python
# データのエンコーディング, マッピングを明示的に
mapping = [
    {"col": "class", "mapping": {"First": 0, "Second": 1, "Third": 2}},
    {"col": "sex", "mapping": {"male": 0, "female": 1}},
    {"col": "embark_town", "mapping": {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}},
    {"col": "deck", "mapping": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}},
]
encoder = IntOrdEncoder(cols=cols, mapping=mapping, handle_unknown='value')
x_tr_enc = encoder.fit_transform(x_tr)

# Catboost
cols = ["class", "sex", "embark_town", "deck"]
clf = CatBoostClassifier(iterations=1000, cat_features=cols)
clf.fit(x_tr_enc, y_tr, verbose=False)
```

---
### Catboostのハイパーパラメータ

* [こちら](https://catboost.ai/en/docs/concepts/parameter-tuning)にCatboostのハイパーパラメータがまとめられている
* 多くのハイパーパラメータがあるが, ここではその中のいくつかをチューニング
  * `depth` - Depth of the tree.
  * `learning_rate` - The learning rate. Used for reducing the gradient step.
  * `random_strength` - The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.
  * `l2_leaf_reg` - Coefficient at the L2 regularization term of the cost function. Any positive value is allowed.

---
## [Optuna](https://www.preferred.jp/ja/projects/optuna/)

* PFNが開発しているOSSのハイパーパラメータ自動最適化フレームワーク
* 以下のようなインターフェースを提供
  * `optuna.Trial` - ハイパーパラメータの分布を定義する機能を提供
  * 目的関数 - `optuna.Trial`を引数に取り目的値を出力する関数
  * `optina.study.Study.optimize()` - 目的関数を最適化

---
## 探索範囲の定義

```python
def suggest_params(trial: optuna.Trial) -> Dict:
    return {
        "depth": trial.suggest_int("depth", 1, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", np.exp(-7.0), 1.0),
        "random_strength": trial.suggest_int("random_strength", 1, 20),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 10),
    }
```

---
## 目的関数の定義

* 与えられたデータに対する目的関数（`objective`）を返す関数

```python
def create_objective(x: DataFrame, y: DataFrame) -> Any:
    (cols, encoder) = titanic_cat_encoder()

    def objective(trial: optuna.Trial) -> Any:
        params = suggest_params(trial)

        clf = CatBoostClassifier(**params, verbose=False)
        pipe = make_pipeline(encoder, clf)　# パイプラインを構成
        score = cross_val_score(pipe, x, y, cv=5).mean() # スコアを計算

        return score

    return objective
```

---
## 最適化の実行

* `optuna.study.Study`オブジェクトを生成し, `optimize()`メソッドを実行

```python
# df_xとdf_yはデータフレーム
objective = create_objective(df_x, df_y)
# 目的関数を「最大化」
study = optuna.create_study(direction='maximize')
# 10最適化ステップ
study.optimize(objective, n_trials=10, show_progress_bar=True)
```

---
## MLflow

* MLライフサイクル管理のためのOSS (DataBricks)
* 構成がシンプルで, 他のOSSとの組み合わせが容易（らしい）
* 以下の4つのコンポーネントを提供

  * MLflow Tracking - 実験管理
  * MLflow Projects - 再現可能なコード
  * MLflow Models - MLモデルの多様な環境へのデプロイ
  * MLflow Model Registry - MLモデルリポジトリ

* ここではMLflow Trackingを使用

---
## MLflow Tracking

* 機械学習の実験パラメータ, メトリクス, 出力ファイル等のロギングと可視化の機能を提供
* 実験を`run`という概念で管理, 各々の`run`は以下の要素を記録：
  * Start and End time, Source, Parameters, Metrics, Artifacts
* 実験結果の保存先としてローカルファイルやDB, クラウドなどを選択可
  * デフォルトではローカルファイルに保存される

---

* `run()`はネスト可能

```python
def fit_eval(x, y, encoder, params, cols, nested):
    with mlflow.start_run(nested=nested):

        # 中略, paramsはハイパーパラメータ, scoreは5-fold cvの結果

        mlflow.log_param("depth", params["depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_metric("cv_score", score)

        return score

paramss = [
  {"depth": 1, "learning_rate": 0.01},
  {"depth": 5, "learning_rate": 0.001},
]

with mlflow.start_run():
    (cols, encoder) = titanic_cat_encoder()

    for params in paramss:
        fit_eval(x_tr, y_tr, encoder, params, cols, nested=True)
```
