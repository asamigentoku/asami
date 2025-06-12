import cv2
import os
import numpy as np
from scipy.ndimage import maximum_filter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def keep_local_max_with_filter(A, a):
    A = A.copy()
    
    # 最大フィルタで近傍の最大値を取得
    max_A=maximum_filter(A, size=(2*a+1,2*a+1), mode='constant')
    #maximum(対象の行列,最大値を出す四角形の大きさ,mode)

    # 元のAと最大値が一致する場所だけ残す（それ以外は0にする）
    A[A < max_A] = 0
    
    return A
a=15

def Wmode(data):
    for x in list(range(len(data))):
        data[x]=int(data[x])
    data_np = np.array(data)
    hist, bin_edges = np.histogram(data_np, bins=30)  # 10等分
    max_bin_index = np.argmax(hist)
    range_start = bin_edges[max_bin_index]
    range_end = bin_edges[max_bin_index + 1]
    print(f"wの値を変化させた上で、一致率が高いものが多いWの値の範囲: {range_start:.1f} ～ {range_end:.1f} に {hist[max_bin_index]} 個")
    super_w=int((range_start+range_end)/2)
    return super_w
    
def Hmode(data):
    for x in list(range(len(data))):
        data[x]=int(data[x])
    data_np = np.array(data)
    hist, bin_edges = np.histogram(data_np, bins=10)  # 10等分この値をdataの大きさに応じて変化させたい
    max_bin_index = np.argmax(hist)
    range_start = bin_edges[max_bin_index]
    range_end = bin_edges[max_bin_index + 1]
    print(f"要素が多く含まれる高さの値の範囲: {range_start:.1f} ～ {range_end:.1f} に {hist[max_bin_index]} 個")
    u=[0,0,0,0]
    u[0]=int((range_start+range_end)/2)
    u[1]=int(range_start-range_end/2)
    u[2]=int(range_start)
    u[3]=int(range_end)
    return u
    



#読み込んだ画像の上の部分は切り取ってもいい
#またハイの形から認識をしていく
#そしてそこから上の部分や下の部分は削ぎ落とす


img = cv2.imread('/Users/asamigentoku/Desktop/2年生前期/プロジェクト高村/麻雀得点計算/対局画像.jpg', cv2.IMREAD_COLOR)
h, w = img.shape[:2]
while True:
    if h<2000:
        break
    img = cv2.resize(img,(int(w*0.65),int(h*0.65)))  # (幅, 高さ) の順番に注意！
    h=int(h*0.65)
    w=int(w*0.65)
res_rows=h
res_cols=w
#ファイル操作
files = os.listdir('/Users/asamigentoku/Desktop/2年生前期/プロジェクト高村/麻雀得点計算/リサイズ')
print(files)
ze=[]
for x in files:
    ze.append(os.path.abspath(os.path.join('/Users/asamigentoku/Desktop/2年生前期/プロジェクト高村/麻雀得点計算/リサイズ', x)))

del ze[0]
print(ze)

all_maxval=[]
all_name=[]
all_w=[]
all_h=[]

all_template_box=[]

super_num_box=[]
for source in ze:
    template = cv2.imread(source, cv2.IMREAD_COLOR)
    w, h = template.shape[1], template.shape[0]
    
    #ファイル名からscannameのリストのall_namwを作る
    name=list(source)
    name.reverse()
    scanname=(name[5]+name[4])
    print(scanname)
    all_name.append(scanname)
    
    
    
    box=[]
    resbox=[]
    resmax=0
    w_box=[]
    h_box=[]
    template_box=[]
    
    #画像データを小さくリサイズし、一致率のmax値の値を並べて、テンプレートマッチングに使う大きさを決定
    for x in list(range(6)):
        if x>0:
            template=cv2.resize(template,(int(w*0.8),int(h*0.8)))  # (幅, 高さ) の順番に注意！
            h=h*0.8
            w=w*0.8
            res = cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)
            box.append(res.max())
            H, W = res.shape
            #res[:int(H*1/3):,:]=0
            #res[-int(H*1/3):,:]=0
            resbox.append(res)
            h_box.append(h)
            w_box.append(w)
            template_box.append(template)
        elif x==0:
            res = cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)
            box.append(res.max())
            H, W = res.shape
            #res[:int(H*1/3):,:]=0
            #res[-int(H*1/3):,:]=0
            resbox.append(res)
            h_box.append(h)
            w_box.append(w)
            template_box.append(template)
    all_template_box.append(template_box)


    max_val=None
    
    #resmaxの値によって、numを選定し、その時の一致率をリストにする--
    for x in list(range(1,len(box)-1)):
        if box[x]>box[x-1] and box[x]>box[x+1]:
            max_val=box[x]
            num=x
        break
    if max_val==None:
        if box[0]>box[1]:
            max_val=box[0]
            num=0
        else:
            max_val=max(box)
            num=box.index(max_val)
    print(max_val)
    all_maxval.append(max_val)
    #------------------------------
    
    #一定の一致率以外は削除したい
    if max_val<0.20:
        break
    res_max=resbox[num]
    h=h_box[num]
    w=w_box[num]
    
    super_num_box.append(num)
    
    all_w.append(w)
    all_h.append(h)
    
print(all_maxval)
print(all_w)
print(f"{len(all_maxval)}と{len(all_w)}")

#決定された一致率の上位20位のwの値をリストにする
top20 = sorted(all_maxval,reverse=True)[:20]
top20_w=[]
for x in top20:
    top20_w.append(all_w[all_maxval.index(x)])

print(top20_w)



print(f"最大スコアは{max(all_maxval)}でありそのパイは{all_name[all_maxval.index(max(all_maxval))]}でありそのハイのおおきさは{all_w[all_maxval.index(max(all_maxval))]}")

#top20個のwのリストからグラフにより、最も良いsuper_wを選ぶ
super_w=Wmode(top20_w)
super_h=super_w*(4/3)



new_res = np.zeros((res_rows, res_cols),dtype=float)
new_res1=np.zeros((res_rows, res_cols),dtype=float)
new_resname = np.full((res_rows, res_cols),'0', dtype='<U10')
new_resname1= np.full((res_rows, res_cols),'0', dtype='<U10')





for source in ze:
    template = cv2.imread(source, cv2.IMREAD_COLOR)
    w, h = template.shape[1], template.shape[0]
    
    name=list(source)
    name.reverse()
    scanname=(name[5]+name[4])
    print(scanname)
    all_name.append(scanname)
    
    box=[]
    resbox=[]
    resmax=0
    w_box=[]
    h_box=[]
    template_box=[]
    for x in list(range(6)):
        if x>0:
            template=cv2.resize(template,(int(w*0.8),int(h*0.8)))  # (幅, 高さ) の順番に注意！
            h=h*0.8
            w=w*0.8
            res = cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)
            box.append(res.max())
            resbox.append(res)
            h_box.append(h)
            w_box.append(w)
            template_box.append(template)
        elif x==0:
            res = cv2.matchTemplate(img,template, cv2.TM_CCOEFF_NORMED)
            box.append(res.max())
            resbox.append(res)
            h_box.append(h)
            w_box.append(w)
            template_box.append(template)


    max_val=None
    for x in list(range(1,len(box)-1)):
        if box[x]>box[x-1] and box[x]>box[x+1]:
            max_val=box[x]
            num=x
        break
    if max_val==None:
        if box[0]>box[1]:
            max_val=box[0]
            num=0
        else:
            max_val=max(box)
            num=box.index(max_val)
    
    #一定の一致率以外は削除したい
    if max_val<0.10:
        break
    res_max=resbox[num]
    h=h_box[num]
    w=w_box[num]
    
    
    
    res_max[res_max<0]=0
    #比率が決まったら横画像の時の認識率をみたい
    a=int(super_w*0.8)
    res_max=keep_local_max_with_filter(res_max,a)
    #４つ以上読み取ったら削除したい
    threshold =res_max.max()-0.10
    loc = np.where(res_max >= threshold)
    #count=np.sum(res_max >= threshold)
    for pt in zip(*loc[::-1]):  # x, yの順に変換
        if w>super_w-20:
            x,y=pt
            new_res[y][x]=res_max[y][x]
            new_resname[y][x]=scanname

new_res=keep_local_max_with_filter(new_res,a)


#上下1/4を削除
#H, W = new_res.shape
#new_res[:int(H*1/4):,:]=0
#new_res[-int(H*1/4):,:]=0

min_h=[]
min_res = []

new_loc= np.where(new_res>0)
for new_pt in zip(*new_loc[::-1]):  # x, yの順に変換
    x,y=new_pt
    min_h.append(y)
    min_res.append(new_res[y][x])

hmode=Hmode(min_h)
a=int(hmode[0]-hmode[1]/2)
b=int(hmode[0]+super_h)
new_res[:a,:]=0
new_res[b:,:]=0

min_res_max=max(min_res)
min_h_max=min_h[min_res.index(min_res_max)]
print(f"この長方形に区切った中で最もスコアが高い時の高さは{min_h_max}です")

#super_wによって決められたteplateを利用してマッチングをする
Tem=all_template_box[0] 

numheikin=sum(super_num_box)/len(super_num_box)
tem_num=round(numheikin)
#上によってtemの大きさのインデックスが決定した


print(all_name)

#teplateマッチングを再び行う
for tem1_val, tem1 in enumerate(all_template_box):
    tem_num=super_num_box[tem1_val]
    tem2=tem1[tem_num]
    res=cv2.matchTemplate(img,tem2, cv2.TM_CCOEFF_NORMED)
    #フィルタリング  
    or_loc= np.where(res>0)
    for or_pt in zip(*or_loc[::-1]):
        x,y=or_pt
        if res[y][x]>new_res1[y][x]:
            new_res1[y][x]=res[y][x]
            new_resname1[y][x]=all_name[tem1_val]
            print(f"この認識は{all_name[tem1_val]}")

#特定の行だけを抽出
hmode=Hmode(min_h)
res_filtered=np.zeros_like(new_res1)         # すべて0に初期化
res_filtered[hmode[2]:hmode[3],:]=new_res1[hmode[2]:hmode[3],:]   
new_res1=res_filtered

#-------------------------------
col_sums=np.sum(new_res1, axis=0)
plt.bar(range(new_res1.shape[1]), col_sums)
plt.xlabel("列の番号")
plt.ylabel("合計値")
plt.title("各列の合計値")
plt.xticks(range(new_res1.shape[1]), [f"col {i}" for i in range(new_res1.shape[1])])
plt.show()
print(col_sums)
peaks, _ = find_peaks(col_sums,prominence=3,distance=super_w*0.7)#山の高さは変わるので注意
peak_list = peaks.tolist()
print("ピークのインデックス:", peaks)
print("ピークの値:", col_sums[peaks])
#------------------------------------

waku=peak_list
    
#------------------------------------------------
model = tf.keras.models.load_model('/Users/asamigentoku/Desktop/2年生前期/プロジェクト高村/機械学習を用いた例/model100.h5')

datagen = ImageDataGenerator(
    rescale=1./255,            # 正規化（ピクセル値を0〜1に）
    rotation_range=20,         # 回転（±20度）
    width_shift_range=0.1,     # 横方向に最大10%ずらす
    height_shift_range=0.1,    # 縦方向に最大10%ずらす
    zoom_range=0.2,            # 拡大・縮小（±20%）
    horizontal_flip=True,      # 左右反転
    validation_split=0.2       # 20%を検証用に使う
)

train_gen = datagen.flow_from_directory(
    '/Users/asamigentoku/Desktop/2年生前期/プロジェクト高村/機械学習を用いた例/mahjong_tiles2',
    target_size=(600,420),        # 画像サイズを統一　　全てのサイズが64*64にリサイズされるということ
    batch_size=32,               # 一度に読み込む枚数
    class_mode='categorical',    # 多クラス分類
    subset='training'            # 学習用
)

h=res_rows
w=res_cols

#配列を初期化
new_res2=np.zeros((res_rows, res_cols),dtype=float)
new_resname2= np.full((res_rows, res_cols),'0', dtype='<U10')


for w1 in waku:
    for h1 in list(range(hmode[3],hmode[3]+1)):
        treimg=img[h1:h1+int(super_h),w1:w1+int(super_w)]
        treimg=cv2.cvtColor(treimg, cv2.COLOR_BGR2RGB)
        treimg = cv2.resize(treimg, (600, 420))
        img_array = np.expand_dims(treimg,axis=0)
        # ピクセル値を0〜1に正規化（学習時と同じように）
        
        print(f"今の進捗は{w1}/{w-super_w}で{h1%hmode[2]}/{hmode[3]-hmode[2]}")
        
        img_array=img_array.astype('float32')
        img_array /= 255.0
        # 予測（確率の配列）
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        print(f"一致率は{max(predictions[0])}")
        # クラス名（train_gen.class_indices は {"class_name": index} の辞書）
        class_indices = train_gen.class_indices
        # index -> class_name に変換
        class_names = {v: k for k, v in class_indices.items()}
        print("この画像は「{}」と予測されました。".format(class_names[predicted_class]))
        new_res2[h1][w1]=max(predictions[0])
        new_resname2[h1][w1]=class_names[predicted_class]
        
#--------------------------------------------------------------------

a=int(super_w*0.8)
new_res2=keep_local_max_with_filter(new_res2,a)


new_loc= np.where(new_res2>0)
for new_pt in zip(*new_loc[::-1]):  # x, yの順に変換
    x,y=new_pt
    new_scanname=new_resname2[y][x]
    cv2.rectangle(img, new_pt, (int(new_pt[0]+super_w), int(new_pt[1]+super_h)), (0, 255, 0), 2)
    cv2.putText(img,new_scanname,new_pt,cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2, cv2.LINE_AA) 




cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
