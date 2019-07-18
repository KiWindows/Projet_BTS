import cv2
import numpy as np
import tkinter

import numpy as np
import tensorflow as tf

drawing = False
draw_size = 500
shift_line = int(draw_size/50)
radius_circle = int(shift_line/2)

img = np.zeros((draw_size, draw_size), np.uint8)

model_path = "model/model.h5"
new_model = tf.keras.models.load_model(model_path)
new_model.summary()

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def interactive_drawing(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        cv2.circle(img, (x, y), radius_circle, (255), -1)
        cv2.imshow('drawingArea', img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x, y), (255), shift_line)
            ix, iy = x, y
            cv2.imshow('drawingArea', img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        predict_button()


def predict_button():
    global img, labels
    x = np.zeros((1, 45, 45), dtype='float64')
    new_img = cv2.resize(img, (45, 45))

    thresh = 127
    im_bw = cv2.threshold(new_img, thresh, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow('predicted', new_img)
    x[0] = np.array(new_img)
    result = new_model.predict_on_batch(x)

    print(result[0])
    r, l = zip(*sorted(zip(result[0], labels), reverse=True))
    show_predicted(result[0])
    print(l)
    print(r)

    print("###################END###################")


def clear_button():
    global img
    img = np.zeros((draw_size, draw_size), np.uint8)
    cv2.imshow('drawingArea', img)


def show_predicted(result):
    global labels
    res = sorted(zip(result, labels), reverse=True)
    text = ""
    for r, l in res:
        text = text + (str(l) + " : " + str(round((r*100), 2)) + "\n")
    lbl.configure(text=text)


window = tkinter.Tk()
btn_predict = tkinter.Button(
    window, text="predict", width=10, height=5, font=("Arial Bold", 25), command=predict_button)


btn_clear = tkinter.Button(
    window, text="clear", width=10, height=5, font=("Arial Bold", 25), command=clear_button)


btn_predict.grid(column=0, row=0)
btn_clear.grid(column=0, row=1)

lbl = tkinter.Label(window, text="- Result -", font=("Arial Bold", 25))

lbl.grid(column=1, row=0, rowspan=2, columnspan=2)


cv2.namedWindow('drawingArea')
cv2.setMouseCallback('drawingArea', interactive_drawing)
cv2.imshow('drawingArea', img)


window.mainloop()
