import dlib
import cv2
import imutils
import sys

def detect(img_name):
  # 讀取照片圖檔
  img = cv2.imread(img_name)

  # 縮小圖片
  img = imutils.resize(img, width=1280)

  # Dlib 的人臉偵測器
  detector = dlib.get_frontal_face_detector()

  # 偵測人臉
  face_rects = detector(img, 0)
  faces = 0;
  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 4, cv2.LINE_AA)
    faces += 1;

  if faces == 1:
    text = str(faces) + " face detected"
    print(faces,"face Detected.")
  else:
    text = str(faces) + " faces detected"
    print(faces,"faces Detected.")

  # 圖片/起點座標/對向座標/顏色/粗細
  cv2.rectangle(img,(80,40),(700,120),(255,255,255),-1)
  # 照片/添加的文字/左上角坐標/字體/字體大小/顏色/字體粗細
  cv2.putText(img,text,(100,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),5)

  # 顯示結果
  cv2.imshow("Face Detection", img)
  negative_imageName = "NoNegative_" + img_name
  cv2.imwrite(negative_imageName,img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
# Main
img = sys.argv[1]
detect(img)

