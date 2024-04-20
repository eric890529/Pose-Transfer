import cv2

img1 = cv2.imread('./grid/model_10/attnmap_5.png')

img2 = cv2.imread('./ori/model_10/attnmap_5.png')
print(img1.shape)
source = img1[0:256, 0:256]

attn1 = img1[256:,256:]

attn2 = img2[256:,256:]

cv2.imwrite('./chooseModel/source.jpg', source)
cv2.imwrite('./chooseModel/attn1.jpg', attn1)
cv2.imwrite('./chooseModel/attn2.jpg', attn2)
