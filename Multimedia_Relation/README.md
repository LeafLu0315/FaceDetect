# 多媒體系統 期末專案

# Multimedia_Relation

  可以辨識照片/影片中的人像，利用 OpenCV + Dlib 函式庫

  FaceDetect_Pic_Negative.py:會將模型中識別為"不像人像"框起

  FaceDetect_Video.py:將影片切為20fps 以影格(frame)方式 識別人像

### 安裝內容:
  * python 3.6
    * imutils
  * dlib 19.6.1
  * opencv

### 使用說明
  安裝Python後
  >python FaceDetect_Pic_Negative.py <圖片>

  >python FaceDetect_Pic_noNegative.py <圖片>

  >python FaceDetect_Video.py <來源影片> <輸出影片>
