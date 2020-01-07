# Project5-5
# 保有隱私之機器學習
[GitHub](https://github.com/cislab-yzu/Project5-5_Open)
## 分工表
| 學號姓名 | 貢獻度 |
| -------- | -------- | 
| 1051446 游采蓉     | 20%   | 
| 1051507 林益聖     | 20%    |
| 1053325 鐘偉豪     | 40%    | 
| 1053349 林剛煇     | 20%    | 

---

## 概念
現今需多電腦的運算都會丟到雲端來做，這時資料的隱私就顯得相當重要，因此在這個專案中，我們利用了同態加密的方法事先對我們資料進行了加密，之後再將這些資料去進行運算，也就是訓練我們機器學習所要用的Model
## 同態加密
https://python-paillier.readthedocs.io/en/stable/phe.html
簡單來說就是加密後的數值經過運算之後，將其解密會跟正確答案一樣
```
舉例來說
12 + 18 = 30
將12與18分別加密之後變成0x123456與0x654321
0x123456 + 0x654321 = 0x456123
將 0x456123解密後會得到30
```
## 流程
1. 資料進行前處理、特徵工程等
2. 將處裡後的資料進行加密
3. 使用加密後的資料去進行模型的訓練
4. 將test data進行加密
5. 使用我們訓練的模型以及加密後的test data預測結果
6. 將結果解密並與ground truth比對
## 驗證
我們在這個專案所使用的資料為Kaggle上經典的house price房價預測資料
使用的機器學習方法為linear regression，過程中並沒有套用現成的機器學習套件，因為同態加密後的運算有一定的限制，像是乘法只能加密的數值乘上未加密的數值，此外也不能比大小即大於小於，因此我們寫了一個專門用在同態加密的linear regression程式。
我們抓了小數點後6位來進行比較，加密與未加密的模型出來的結果完全一樣，至於抓後6位的原因是python在小數點後10位左右會開始有誤差，因此我們只選擇到後6位進行比較。

## Linear Regression
### Function
1.	dot()
將兩個向量做內積，回傳內積的值。舉例來說:
w:	[1,2,3]
d:	[4,5,6]
回傳  w[0]*d[0] + w[1]*d[1] + w[2]*d[2] = 4 + 10 + 18 = 32
2.	normalize()
如果沒有將資料做normalize的話會導致後續在做Dot的時候出現overflow的情形，因此之後預測出來的數值也會是normalize之後的值，因此要再做處裡才會是我要提交的值。公式為: (X-min)/(max-min)
3.	loss()
這個function會回傳gradient給linear_regression()，也就是負責計算SSE等數值的地方，與講義上所做的事情是一樣的，因此就不細說了。
4.	linear_regression()
這個function很簡單，就是不斷呼叫loss()來計算gradient，並依照算出來的gradient來改變theta值，我設定的停止條件是跑到設定的iteration就停止。
5.	predict()
用來預測test data的function，會回傳一個list，內容為預測的房價的normalize數值，在main()還要做處理，才會是最終的結果。
### 資料前處理
#### 讀取資料
我先用pandas將csv檔讀進來，這是因為如果用numpy的話會無法處理內容為字串的資料。
資料讀進來之後，先將train data中的”SalePrice”這一欄位切割出來，因為這是我們要預測的欄位，也是訓練模型時要做比對的欄位，理所當然的不能拿來當作特徵，我將切割出來的資料命名為result。
取特徵:
再來就是提取特徵的部分了，扣除SalePrice後有79項特徵，根據特徵重要度來篩選要使用的特徵。
#### 補缺失值
將特徵提取完之後就是補上缺失值，我使用pandas內建的fillna()，這個function可以將NaN的變成我指定的值，這次我是直接將NaN補成0。這一部分我認為還有很大的改進空間。
	將字串轉換成int:
		程式碼如下:
		for col in test_data:
   			 if (test_data[col].dtypes == 'object'):
     			   test_data[col] = pd.factorize(test_data[col], sort= True)[0]
這個for loop會將每個非數值的列做轉換，轉換的方式為計算這個column有幾項不同的值，並先將各個不同的值做排序後依小到大賦予其數值。舉例來說，[‘apple’, ‘banana’, ‘cat’]總共有3種類別，因此就會轉換成[1,2,3]。
這邊有一點要注意，train data跟test data要先合併處理，這是因為有可能train跟test的資料不相同，舉例來說train為[‘apple’, ‘banana’, ’orange’]但是test是[‘apple’, ‘banana’, ’water melon’]，這邊會出現一個問題，被分類到3的分別是orange跟water melon，這在訓練的時候會造成極大錯誤，因此要先合併再分類。Normalize也是一樣的情況。
#### 將pandas轉換成陣列
經過以上的處理之後，所有的值都已數值化，因此我將pandas轉換成numpy的形式，因為這樣我會比較方便提取、計算其中的數值，因為使用pandas在提取、計算數值方面有許多的限制，嚴重一點甚至會導致資料的內容錯誤，因此我先將panda -> numpy.array以確保不會出現上述情形，語法很簡單，以train_data為例: train_data = train_data.values。
#### 將資料加密
使用paillier將資料進行同態加密
#### 模型訓練&預測
完成資料處理之後就是呼叫linear_regression()來訓練我們的模型，在這邊用weight來代表訓練好的模型，訓練完之後就是使用該模型來預測test data，這邊要注意一件事，test data要與train data做一樣的處理，也就是上述的取特徵、補缺失值、字串轉int、pandas轉換成陣列。
得到weight後，將test data與weight當作參數丟給predict()就可以得到房價的預測結果，不過這邊要注意一件事，我必須逆normalize，前面有寫到normalize的公式為(X-min)/(max-min) = X’，因此X’要推回X的話就要 X’(max-min) + min，這樣就可以推回X。
