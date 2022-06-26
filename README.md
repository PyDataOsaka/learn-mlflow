# learn-mlflow

このリポジトリにはMLFlowを用いてハイパーパラメータ探索を実行する手順を説明するためのノートブックが含まれます.

TODO: なぜMLFlow + Optunaか. 後で書く.

以下のノートブックがあります.

* `catencoder.ipynb` - TitanicデータセットにCatboostを適用します. Scikit-learnのパイプラインを使用します.
* `optuna.ipynb` - CatboostのハイパーパラメータをOptunaで最適化します.
* `mlflow-tracking.ipynb` - MLFlowを用いて実験結果を記録します.
* `mlflow-optuna.ipynb` - MLFlowを用いてOptunaによるハイパーパラメータ最適化の結果を記録します.

## 実行方法

* git clone
* pyenvで仮想環境を作成
* poetry install
* ノートブックを実行

（後でもっとちゃんと書く）

## その他

* [自分用の作業ログ](https://hackmd.io/qDRLJFBxTcCavCPZZ0KZ9A)（他の人には見えないはず, 見えても困りませんが）
