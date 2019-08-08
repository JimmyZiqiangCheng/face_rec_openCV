import cv2

# read image as b&w
img_bw = cv2.imread("sample.jpg", 0)
# resize
img_bw_resize = cv2.resize(img_bw, (int(img_bw.shape[1]/4),int(img_bw.shape[0]/4)))

# read image as colored
img = cv2.imread("sample.jpg")
# resize
# grey scale img
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cascade classifier object for face recognition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# find the faces
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.18, minNeighbors=5)
# create rectangle to highlight faces
for x, y, w, h in faces:
	img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 5)

img_resize = cv2.resize(img, (int(img.shape[1]/4),int(img.shape[0]/4)))

cv2.imshow("sample_image", img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()