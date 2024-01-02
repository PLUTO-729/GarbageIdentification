import argparse
import datetime
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer, QDateTime, QDate, QTime, QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from torch.backends import cudnn

from models.experimental import attempt_load
from ui.mainwindow_ui import Ui_MainWindow
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box2


class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.initUI()

    # 初始化界面
    def initUI(self):
        self.setWindowIcon(QIcon("./icon/yolo.png"))
        # 创建一个窗口对象
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer_video = QtCore.QTimer(self) # 创建定时器
        self.timer_photo = QtCore.QTimer(self) # 创建定时器
        self.output_folder = 'output/'
        self.cap = cv2.VideoCapture()
        self.vid_writer = None
        self.camera_detect = False
        self.num_stop = 1  # 暂停与播放辅助信号，note：通过奇偶来控制暂停与播放
        self.openfile_name_model = None        # 权重初始文件名
        self.count = 0
        self.start_time = time.time()        # 打开线程
        self.stop_going = 0



        # 刷新lcd时间
        self.lcd_time = QTimer(self)
        self.lcd_time.setInterval(1000)
        self.lcd_time.timeout.connect(self.refresh)
        self.lcd_time.start()

        self.ui.textBrowser_print.append("特别说明：如需启动检测，请先加载weights文件！！！")
        self.init_slots()

    # 刷新时间
    def refresh(self):
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.lcdNumber.display(now_time)

    # 初始化槽函数
    def init_slots(self):
        self.ui.btn_loadweight.clicked.connect(self.load_model)
        self.ui.btn_loadimg.clicked.connect(self.button_image_open)
        self.ui.btn_loadvideo.clicked.connect(self.button_video_open)
        self.ui.btn_opencamera.clicked.connect(self.button_camera_open)
        self.ui.btn_camera_detect.clicked.connect(self.button_camera_detect)
        self.ui.btn_stop.clicked.connect(self.button_stop)
        self.ui.btn_over.clicked.connect(self.button_over)
        self.ui.btn_closecamera.clicked.connect(self.button_closecamera)
        self.ui.btn_clear.clicked.connect(self.button_clear)
        self.ui.btn_takephoto.clicked.connect(self.button_takephoto)
        self.ui.btn_labelimg.clicked.connect(self.button_labelimg)

        self.timer_video.timeout.connect(self.show_video_frame)  # 定时器超时，将槽绑定至show_video_frame
        self.timer_photo.timeout.connect(self.show_image)  # 定时器超时，将槽绑定至show_video_frame


    # 加载模型
    def load_model(self):
        self.openfile_name_model, _ = QtWidgets.QFileDialog.getOpenFileName(self.ui.btn_loadweight, '选择weights文件',
                                                                  'weights/', "*.pt;;*.pth")
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.textBrowser_print.append("打开权重失败")
        else:
            self.ui.textBrowser_print.append("加载weights文件地址为：" + str(self.openfile_name_model))
            self.model_init() #初始化权重

    # 初始化权重
    def model_init(self):
        # 模型相关参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s6.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        # 改变权重文件
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print(weights)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        #  Second-stage classifier

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # 设置提示框
        QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        self.ui.textBrowser_print.append("模型加载完成")

    # 打开图片
    def button_image_open(self):
            # 打印信息显示在界面
            name_list = []
            try:
                img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/images", "*.jpg;;*.png;;All Files(*)")
            except OSError as reason:
                print('文件打开出错啦！核对路径是否正确'+ str(reason))
                self.ui.textBrowser_print.append("文件打开出错啦！核对路径是否正确")
            else:
                # 判断图片是否为空
                if not img_name:
                    QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                                  defaultButton=QtWidgets.QMessageBox.Ok)
                    self.ui.textBrowser_print.append("打开图片失败")
                else:
                    self.ui.textBrowser_print.append("打开图片成功")
                    img = cv2.imread(img_name)
                    print("img_name:", img_name)
                    self.origin = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self.origin = cv2.resize(self.origin, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg_origin = QtGui.QImage(self.origin.data, self.origin.shape[1], self.origin.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.ui.label_origin.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_origin))
                    self.ui.label_origin.setScaledContents(True)  # 设置图像自适应界面大小

                    info_show = self.detect(name_list, img)
                    # print(info_show)
                    # 获取当前系统时间，作为img文件名
                    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                    file_extension = img_name.split('.')[-1]
                    new_filename = now + '.' + file_extension  # 获得文件后缀名
                    file_path = self.output_folder + 'img_output/' + new_filename
                    cv2.imwrite(file_path, img)
                    # 检测信息显示在界面
                    self.ui.textBrowser_detect.append(info_show)

                    # 检测结果显示在界面
                    self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
                    self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                              QtGui.QImage.Format_RGB32)
                    self.ui.label_detect.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.ui.label_detect.setScaledContents(True)  # 设置图像自适应界面大小

    # 目标检测
    def detect(self, name_list, img):
        '''
           :param name_list: 文件名列表
           :param img: 待检测图片
           :return: info_show:检测输出的文字信息
        '''
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            info_show = ""
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        single_info = plot_one_box2(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                    line_thickness=2)
                        info_show = info_show + single_info + "\n"
        return info_show


    # 打开视频并检测
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "data/video/", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)

        # 判断摄像头是否打开
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # -------------------------写入视频----------------------------------#
            self.ui.textBrowser_print.append("打开视频检测")
            fps, w, h, save_path = self.set_video_name_and_path()
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            self.timer_video.start(30)  # 以30ms为间隔，启动或重启定时器
            # 进行视频识别时，关闭其他按键点击功能
            self.ui.btn_loadvideo.setDisabled(True)
            self.ui.btn_loadimg.setDisabled(True)
            self.ui.btn_opencamera.setDisabled(True)

    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path

    # 定义视频帧显示操作
    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()

        # 显示视频数据的帧数
        self.count += 1
        if self.count % 10 == 0:
            self.count = 0
            fps = int(30 / (time.time() - self.start_time))
            self.ui.fps_label.setText('fps:' + str(fps))
            self.start_time = time.time()

        if img is not None:
            # 原始数据的显示
            self.origin = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.origin = cv2.resize(self.origin, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg_origin = QtGui.QImage(self.origin.data, self.origin.shape[1], self.origin.shape[0],
                                             QtGui.QImage.Format_RGB32)
            self.ui.label_origin.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_origin))
            self.ui.label_origin.setScaledContents(True)  # 设置图像自适应界面大小

            # 检测数据的显示
            info_show = self.detect(name_list, img)  # 检测结果写入到原始img上
            self.vid_writer.write(img)  # 检测结果写入视频
            print(info_show)
            # 检测信息显示在界面
            self.ui.textBrowser_detect.append(info_show)
            show = cv2.resize(img, (640, 480))  # 直接将原始img上的检测结果进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.ui.label_detect.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.label_detect.setScaledContents(True)  # 设置图像自适应界面大小
        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            self.cap.release() # 释放video_capture资源
            self.vid_writer.release() # 释放video_writer资源
            self.ui.label.clear()
            # 视频帧显示期间，禁用其他检测按键功能
            self.ui.btn_loadvideo.setDisabled(True)
            self.ui.btn_loadimg.setDisabled(True)
            self.ui.btn_opencamera.setDisabled(True)

    '''显示图片'''
    def show_image(self):
        flag, self.image = self.cap.read()  # 从视频流中读取图片
        image_show = cv2.resize(self.image, (620, 420))  # 把读到的帧的大小重新设置为显示的窗口大小
        width, height = image_show.shape[:2]  # 行:宽，列:高
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)  # opencv读的通道是BGR,要转成RGB
        image_show = cv2.flip(image_show, 1)  # 水平翻转，因为摄像头拍的是镜像的。
        # 把读取到的视频数据变成QImage形式(图片数据、高、宽、RGB颜色空间，三个通道各有2**8=256种颜色)
        self.photo= QtGui.QImage(image_show.data, height, width, QImage.Format_RGB888)
        self.ui.label_origin.setPixmap(QPixmap.fromImage(self.photo))  # 往显示视频的Label里显示QImage
        self.ui.label_origin.setScaledContents(True)  # 图片自适应

    # 使用摄像头检测
    def button_camera_open(self):
        self.camera_detect = True
        self.ui.textBrowser_print.append("打开摄像头")
        # 设置使用的摄像头序号，系统自带为0
        camera_num = 0
        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_num)
        # 判断摄像头是否处于打开状态
        bool_open = self.cap.isOpened()
        if not bool_open:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.information(self, u"Warning", u"打开摄像头成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.ui.btn_loadvideo.setDisabled(True)
            self.ui.btn_loadimg.setDisabled(True)

    # 启动摄像头检测
    def button_camera_detect(self):
        self.ui.textBrowser_print.append("启动摄像头检测")
        fps, w, h, save_path = self.set_video_name_and_path()
        fps = 5  # 控制摄像头检测下的fps，Note：保存的视频，播放速度有点快，我只是粗暴的调整了FPS
        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        self.timer_video.start(30)
        self.ui.btn_loadvideo.setDisabled(True)
        self.ui.btn_loadimg.setDisabled(True)
        self.ui.btn_opencamera.setDisabled(True)


    # 视频暂停按钮
    def button_stop(self):

        self.timer_video.blockSignals(False)
        # 暂停检测
        # 若QTimer已经触发，且激活
        if self.timer_video.isActive() == True and self.num_stop % 2 == 1:
            self.ui.btn_stop.setText('继续')
            self.ui.textBrowser_print.append("视频暂停播放")
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer_video.blockSignals(True)
            # 继续检测
        else:
            self.num_stop = self.num_stop + 1
            self.ui.btn_stop.setText('暂停')
            self.ui.textBrowser_print.append("视频继续播放")

    # 停止视频播放
    def button_over(self):
        self.ui.textBrowser_print.append("视频结束播放")
        self.cap.release()  # 释放video_capture资源
        self.timer_video.stop()  # 停止读取
        self.timer_photo.stop()  # 停止读取
        if self.vid_writer != None:
            self.vid_writer.release()  # 释放video_writer资源

        self.ui.label_origin.clear()  # 清空label画布
        self.ui.label_detect.clear()  # 清空label画布
        # 启动其他检测按键功能
        self.ui.btn_loadvideo.setDisabled(False)
        self.ui.btn_loadimg.setDisabled(False)
        self.ui.btn_opencamera.setDisabled(False)

        # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
        # Note:点击暂停之后，num_stop为偶数状态
        if self.num_stop % 2 == 0:
            print("Reset stop/begin!")
            self.ui.btn_stop.setText(u'暂停')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)

    # 关闭摄像头
    def button_closecamera(self):
        self.ui.textBrowser_print.append("关闭摄像头")
        self.ui.fps_label.setText("帧率")
        self.timer_video.stop()  # 停止读取
        self.timer_photo.stop()  # 停止读取
        self.cap.release()  # 释放摄像头
        self.ui.label_origin.clear()  # 清空label画布
        self.ui.label_detect.clear()  # 清空label画布
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 摄像头

        self.ui.btn_loadvideo.setDisabled(False)
        self.ui.btn_loadimg.setDisabled(False)
        self.ui.btn_opencamera.setDisabled(False)

    # 拍照
    def button_takephoto(self):
        self.ui.textBrowser_print.append("启动拍照")
        self.timer_photo.start(30)
        self.show_image()
        if self.cap.isOpened():
            FName = "data/images" + fr"/img{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
            print(FName)
            # 原始数据的显示
            flag, self.image = self.cap.read()  # 从视频流中读取图片
            image_show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为显示的窗口大小
            image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)  # opencv读的通道是BGR,要转成RGB
            image_show = cv2.flip(image_show, 1)  # 水平翻转，因为摄像头拍的是镜像的。
            # 把读取到的视频数据变成QImage形式(图片数据、高、宽、RGB颜色空间，三个通道各有2**8=256种颜色)
            self.showImage = QtGui.QImage(image_show.data, image_show.shape[1], image_show.shape[0], QImage.Format_RGB888)
            self.ui.label_detect.setPixmap(QtGui.QPixmap.fromImage(self.photo))
            self.ui.label_detect.setScaledContents(True)  # 设置图像自适应界面大小
            self.showImage.save(FName + ".jpg", "JPG", 300)
        else:
            QMessageBox.critical(self, '错误', '摄像头未打开！')
            return None

    # 调用lablelimg批注工具
    def button_labelimg(self):
        self.ui.textBrowser_print.append("启动标注工具")
        os.system("labelimg")

    # 清除显示区域
    def button_clear(self):
        self.ui.textBrowser_print.append("清除显示区域")
        self.ui.textBrowser_print.clear()
        self.ui.textBrowser_detect.clear()


    # 窗口居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        # 设置窗口大小
        self.move(qr.topLeft())

    # 关闭事件
    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_video.isActive():
                self.timer_video.stop()
            if self.timer_photo.isActive():
                self.timer_photo.stop()
            event.accept()
        else:
            event.ignore()



if __name__ == '__main__':
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())



