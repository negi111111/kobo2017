# 2017年度 情報工学工房(深層学習班)予定表
- **５限は必ず出席してください**．今後，深層学習フレームワークを用いて実装する上での共通の知識ベースを作ります．
- **６限はなるべく出席することをオススメします**．前半(6月の下旬まで)は主に５限の時間帯は【理論】，６限は【実践(コンテスト対策)】に分けて学習していく予定です．また，７月以降はコンテスト用のチームの話し合いの場とします．
- **夏休み以降は，主にグループ分けをして，各自の趣向に合わせて，応用課題を設定します**．チームまたは個人で最終的に何らかの作品(システム，論文，etc)を作って，調布祭での発表，工房最終発表会の他に，Qiitaなどを使って外部発信を行ってもらいます．
- 後期は，**画像認識/検出/領域分割**，**画像変換/画像生成**，**動画**，**深層強化学習**，**自然言語処理**，**ソーシャルメディア**などに分かれて開発を行います．
- 理論及び実装を突き詰めたい場合は，**深層学習独自フレームワークの作成**もテーマの一つで，その場合は，**Githubの各フレームワークをひたすら読んで理解する**を行うのも一つだと思います．

## 勉強計画(５限)
| 日付 | 5/11 | 5/18 |  5/25 | 6/01 | 6/08 | 6/15 | 6/22 | 6/29 | 7/06 | 7/13 | 7/20 |
|:----:|:----:|:----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 実施 | 予定 | 予定 |  休み | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 |
| 内容 |   0  |   2  |  休み |   2, 3  |   3  |   4  |   5  |   5  |   *  |   *  | * |

## 勉強計画(６限)
| 日付 | 5/11 | 5/18 |  5/25 | 6/01 | 6/08 | 6/15 | 6/22 | 6/29 | 7/06 | 7/13 | 7/20 |
|:----:|:----:|:----:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 実施 | 予定 | 予定 |  休み | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 | 予定 |
| 内容 |   *  |   *  |  休み |   *  |   *  |   *  |   *  |   *  |   *  |   *  | * |
## 深層学習(理論編)：５限実施予定
　ChainerやTensorFlowをはじめ，多くの深層学習フレームワークが公開され，誰でも手軽に利用できるようになっています．なので，フレームワークを用いれば深層学習に関するプログラムを動かすのはそんなには大変ではありません．ですが，フレームワークを利用するとどうしても中身がブラックボックス化してしまい，深層学習の理論を理解せずとも一応は動かせてしまうため，理論が手薄になってしまいがちです．  
　よって，【理論編】においては，フレームワークを用いずに，自分で深層学習に関するプログラムをPythonベースで作り上げることで，ニューラルネットの動作を理解することを目指します．

### 0. イントロダクション
**内容：コース概要説明，自己紹介**

### 1. Numpyの基礎
**内容：Pythonと線形代数，行列・テンソル**  
Pythonの基本的な文法やNumpyなどは座学しないので，不安がある人は以下の資料を参考にして，独自で進めておいてください．  
**参考資料**
- [Dive Into Python 3](http://diveintopython3-ja.rdy.jp/index.html)  フリーでは有名な本．日本語なので気軽に読めます．
- [Linear algebra cheat sheet for deep learning](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)  
機械学習に欠かせない線形代数について、numpyを使いながら基礎的な部分について解説
- [introduction to numpy](https://github.com/rasbt/deep-learning-book/blob/master/code/appendix_f_numpy-intro/appendix_f_numpy-intro.ipynb?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_128)  
「Introduction to Artificial Neural Networks and Deep Learning: A Practical Guide with Applications in Python 」を一通り．ノートブック形式は，[ここ](https://github.com/rasbt/deep-learning-book)
- [Pythonの数値計算ライブラリ NumPy入門](http://qiita.com/wellflat/items/284ecc4116208d155e01)
- [Linear algebra cheat sheet for deep learning](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)

### 2. ニューラルネットワークを用いた手書き文字認識
**内容：パーセプトロン(MLP)，活性化関数，手書き数字分類**  


#### 2.1 パーセプトロン
#### 2.2 活性化関数
#### 2.3 MLP
#### 2.4 手書き数字分類
**資料**
- slide
- notebook

**補足資料**
- [ゼロから作るDeep Learning 2章 パーセプトロン](http://n3104.hatenablog.com/entry/2017/02/08/095940)
- [【jupyterで学ぶ】 ゼロから作るDeep Learning － 第1回：パーセプトロン](http://tawara.hatenablog.com/entry/2016/10/22/DeepLearningFromZero-01)

### 3. ニューラルネットワークの学習
**内容：損失関数，学習アルゴリズムの実装**
#### 3.1 損失関数
#### 3.2 勾配法

**資料**
- slide
- notebook

**補足資料**

### 4. 誤差逆伝播法
#### 4.1 計算グラフ
#### 4.2 逆伝搬
#### 4.3 誤差逆伝播法を使った学習

**資料**
- slide
- notebook

**補足資料**

### 5. Convolutional Neural Networks(CNN)
**内容：畳込み, プーリング**
#### 5.1 全体構造
#### 5.2 畳込み層
#### 5.3 プーリング層

**資料**
- slide
- notebook

**補足資料**

### 6. コンテスト参加会議

### 参考文献
前半の1〜5は[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)を参考にしています．

#### 動画
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)  
シラバスと課題を参考
- [CS224n: Natural Language Processing with Deep Learning
](http://web.stanford.edu/class/cs224n/)  
工房に取り入れたいが，時間がある場合のみ
- [CS228 Python Tutorial](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb)  
PythonとNumpyの基礎に利用
- [Practical Deep Learning For Coders](http://course.fast.ai/)
- [New Course: Deep Learning in Python (first Keras 2.0 online course!)](https://www.datacamp.com/community/blog/new-course-deep-learning-in-python-first-keras-2-0-online-course#gs.RyuZQe0)  
Kerasの学習に利用

## 深層学習(実践編(コンテスト対策))：６限実施予定
５限で学んだ知識をフレームワークを使って実践していきます．今年は，Keras及びChainerを使って，色々な課題を実際に解いてもらいます．また，前期の最終課題として，画像系のコンテストに参加して上位入賞を目指します．

### 1. Kerasチュートリアル
- [Keras Tutorial: Deep Learning in Python
](https://www.datacamp.com/community/tutorials/deep-learning-python#gs.rp6Kgec)にwebで動かせる実行環境があるので，前日までに一通りやって質問対応

### 2. Recognizing cats and dogs
- [Practical Deep Learning For Coders](http://course.fast.ai/)を参考にして行う．

### Chainer
- [Chainerハンズオン](http://qiita.com/mitmul/items/eccf4e0a84cb784ba84a)  
現在，一番良い資料．ノートブック形式は，[ここ](https://github.com/mitmul/chainer-handson)

### Keras

### ChainerとKerasとの違い
- [kerasとchainerの違い](http://s0sem0y.hatenablog.com/entry/2017/01/10/233242)  
**Keras**: TensorFlowをバックエンドに，直感的な記述でニューラルネットを記述可能にしたライブラリ．テンソル計算と計算グラフをTensorFlowで実現して，Kerasはそれをニューラルネット用にまとめたもの．  
**Chainer**: 計算グラフの実装からニューラルネットの学習までの記述をカバー．テンソル計算をCupyやNumpyで補うスタイル．

![フレームワークのソフトウェアスタック](../image/nn.png)

### フレームワーク
- [LSTM で正弦波を予測する](https://intheweb.io/lstm-dezheng-xian-bo-woyu-ce-suru/)  
**Keras**
- [chainerでsin関数を学習させてみた](http://qiita.com/hikobotch/items/018808ef795061176824)  
**Chainer**

- [ディープラーニング実践入門 〜 Kerasライブラリで画像認識をはじめよう！](https://employment.en-japan.com/engineerhub/entry/2017/04/28/110000#2-Keras%E3%81%A7MNIST%E3%81%AE%E6%89%8B%E6%9B%B8%E3%81%8D%E6%95%B0%E5%AD%97%E3%82%92%E8%AA%8D%E8%AD%98%E3%81%95%E3%81%9B%E3%81%A6%E3%81%BF%E3%82%88%E3%81%86)
- [TensorFlow で学ぶディープラーニング入門を読んだ。Kaggle で実践してみた。](http://futurismo.biz/archives/6274)
- [人工知能に関する断創録(Keras)](http://aidiary.hatenablog.com/entry/20170110/1484057655)
- [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- [Chainer: ビギナー向けチュートリアル Vol.1](http://qiita.com/mitmul/items/eccf4e0a84cb784ba84a)
