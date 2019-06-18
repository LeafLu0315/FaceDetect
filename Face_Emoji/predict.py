import cv2
import numpy as np
import sys
from keras.models import model_from_json

model_path = './model/'
img_size = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)

# 從json中加載模型
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# 加載模型權重
model.load_weights(model_path + 'model_weight.h5')

# 加載emotion
emotion_images = {}
for emoji in emotion_labels:
    emotion_images[emoji] = cv2.imread("./emoji/" + emoji + ".png", -1)

# Main Function
def main():
    imageName = sys.argv[1]
    imageCap(imageName)

def face2emoji(face, emotion_index, position):
    x, y, w, h = position
    emotion_image = cv2.resize(emotion_images[emotion_index], (w, h))
    overlay_img = emotion_image[:, :, :3]/255.0
    overlay_bg = emotion_image[:, :, 3:]/255.0
    background = (1.0 - overlay_bg)
    face_part = (face[y:y + h, x:x + w]/255.0) * background
    overlay_part = overlay_img * overlay_bg
    face[y:y + h, x:x + w] = cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)

    return face

def videoCap():
    # 創建VideoCapture對象
    capture = cv2.VideoCapture(0)

    # 使用opencv的人臉分類器
    cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')

    while True:
        ret, frame = capture.read()

        # 灰度化處理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 呈現用emoji替代後的畫面
        emoji_show = frame.copy()

        # 識別人臉位置
        faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=1, minSize=(120, 120))

        if len(faceLands) > 0:
            for faceLand in faceLands:
                x, y, w, h = faceLand
                images = []
                result = np.array([0.0] * num_class)

                # 裁剪出臉部圖像
                image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
                image = image / 255.0
                image = image.reshape(1, img_size, img_size, 1)

                # 調用模型預測情緒
                predict_lists = model.predict_proba(image, batch_size=32, verbose=1)
                # print(predict_lists)
                result += np.array([predict for predict_list in predict_lists
                                    for predict in predict_list])
                # print(result)
                emotion = emotion_labels[int(np.argmax(result))]
                print("Emotion:", emotion)

                emoji = face2emoji(emoji_show, emotion, (x, y, w, h))
                cv2.imshow("Emotion", emoji)

                # 框出臉部並且寫上標籤
                cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20),
                            (0, 255, 255), thickness=10)
                cv2.putText(frame, '%s' % emotion, (x, y - 50),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2, 30)
                cv2.imshow('Face', frame)

            if cv2.waitKey(60) == ord('q'):
                break

    # 釋放攝像頭並銷毀所有窗口
    capture.release()
    cv2.destroyAllWindows()

def imageCap(img_name):
    # read image with gray scale
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用opencv的人臉分類器
    cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_alt.xml')
    # testing write the file
    # cv2.imwrite('test.jpg',img)
    # 呈現用emoji替代後的畫面
    emoji_show = img.copy()
    emotion_count = [0] * num_class
    faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=1)
    if len(faceLands) > 0:
        print("OK")
        for faceLand in faceLands:
            x, y, w, h = faceLand
            images = []
            result = np.array([0.0] * num_class)
            print("IN")
            # 裁剪出臉部圖像
            image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
            image = image / 255.0
            image = image.reshape(1, img_size, img_size, 1)

            # 調用模型預測情緒
            predict_lists = model.predict_proba(image, batch_size=32, verbose=1)
            # print(predict_lists)
            result += np.array([predict for predict_list in predict_lists
                                for predict in predict_list])
            # print(result)
            emotion = emotion_labels[int(np.argmax(result))]
            print("Emotion:", emotion)
            emotion_count[int(np.argmax(result))] += 1

            emoji = face2emoji(emoji_show, emotion, (x, y, w, h))
            cv2.imshow("Emotion", emoji)

            # 框出臉部並且寫上標籤
            # 圖片/起點座標/對向座標/顏色/粗細
            cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20),
                        (255, 255, 0), thickness=2)
            #putText(Mat& img, const string& text, Point org, int fontFace, double fontScale,
            #Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false)
            # 照片/添加的文字/左上角坐標/字體/字體大小/顏色/字體粗細
            cv2.putText(img, '%s' % emotion, (x, y - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, 15)
            cv2.imshow('Face', img)
            if cv2.waitKey(1500) == ord('q'):
                break
        print("total count:")
    for i in range(num_class):
        print(emotion_labels[i],":",emotion_count[i])
if __name__ == "__main__":
    main()