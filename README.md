# learn-mlflow

このリポジトリには[MLFlow](https://mlflow.org/)を学習するためのノートブックが含まれます.
以下の2つのディレクトリが存在します.

* `notebook/optuna` - [Optuna](https://www.preferred.jp/ja/projects/optuna/)を用いて予測モデルのハイパーパラメータ探索を実行し,
  その結果をMLFlowのUIで確認するためのノートブック.
* `notebook/model` - 訓練した予測モデルを前処理を含めて永続化するためのノートブック.

現時点では1つ目のノートブックに対してのみ詳細な説明があります.

## 実行方法

以下の手順で動作確認を行いました.

1. 本リポジトリを`git clone`します.

    ```bash
    git clone https://github.com/PyDataOsaka/learn-mlflow.git
    ```

2. [pyenv](https://github.com/pyenv/pyenv)でPythonのバージョンを3.9.0とします.

    ```bash
    pyenv install 3.9.0
    cd learn-mlflow
    pyenv local 3.9.0
    ```

3. `learn-mlflow`以下に仮想環境を作成します.

    ```bash
    # 場所はlearn-mlflow
    python -m venv venv
    source venv/bin/activate # 仮想環境をアクティベート
    pip install --upgrade pip # pipを最新版にします
    ```

4. 依存ライブラリをインストールします.

    ```bash
    # 場所はlearn-mlflow, 仮想環境をアクティベート済み
    poetry install
    ```

    補足：Apple SiliconのMacでは以下のようにします（[参考](https://github.com/numpy/numpy/issues/17807#issuecomment-731014921)）.

    ```bash
    # brew install openblas でOpenBLASを事前にインストール
    OPENBLAS="$(brew --prefix openblas)" poetry install
    ```

5. Jupyter serverを起動します.

    ```bash
    # 場所はlearn-mlflow, 仮想環境をアクティベート済み
    jupyter lab
    ```

6. ノートブックを実行します. VSCodeのJupyter serverで実行する場合,
  カーネル用のインタプリタが仮想環境のものであることを確認します.

## その他

* [自分用の作業ログ](https://hackmd.io/qDRLJFBxTcCavCPZZ0KZ9A)（他の人には見えないはず, 見えても困りませんが）
