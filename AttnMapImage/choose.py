import cv2
model = 12
num = 0
for num in range(8):
    img1 = cv2.imread('./attnFilter/model_' + str(model)+ '/attnmap_' + str(num) +'.png')

    img2 = cv2.imread('./NoattnFilter/model_' + str(model)+ '/attnmap_' + str(num) +'.png')
    print(img1.shape)

    source = img1[10:642, 0:450]
    source = cv2.resize(source, (176, 256), interpolation=cv2.INTER_AREA)

    attn1 = img1[10:642,500:-10]
    attn1 = cv2.resize(attn1, (176, 256), interpolation=cv2.INTER_AREA)

    attn2 = img2[10:642,500:-10]
    attn2 = cv2.resize(attn2, (176, 256), interpolation=cv2.INTER_AREA)

    cv2.imwrite('./chooseModel/source_' + str(model) + '_num' + str(num) +'.jpg', source)
    cv2.imwrite('./chooseModel/attnFilter_' + str(model)+ '_num' + str(num) + '.jpg', attn1)
    cv2.imwrite('./chooseModel/NoattnFilter_' + str(model)+ '_num' + str(num) + '.jpg', attn2)
