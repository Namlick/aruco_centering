# Copyright (c) farm-ng, inc. Amiga Development Kit License, Version 0.1
import argparse
import asyncio
import os
from typing import List

import cv2
import grpc
import numpy as np
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
from turbojpeg import TurboJPEG

os.environ["KIVY_NO_ARGS"] = "1"


from kivy.config import Config  # noreorder # noqa: E402

Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")  # height in the calibration
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")

from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402


# if ARUCO_DICT.get(args["type"], None) is None: # should test for when the dictionary is invalid
#         print("[INFO] ArUCo tag of '{}' is not supported".format(
#             args["type"]))
#         sys.exit(0)
class MyAruco:
    def __init__(self):
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        }
        self.arucoDict = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT[args.type]
        )  # make so universal
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.detectorObj = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def detect(self, frame):
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str,
            default="DICT_ARUCO_ORIGINAL",
            help="type of ArUCo tag to detect")
        args = vars(ap.parse_args())
        # verify that the supplied ArUCo tag exists and is supported by
        # OpenCV
        if ARUCO_DICT.get(args["type"], None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                args["type"]))
            sys.exit(0)
        # load the ArUCo dictionary and grab the ArUCo parameters   
        print("[INFO] detecting '{}' tags...".format(args["type"]))
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
        arucoParams = cv2.aruco.DetectorParameters()
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        #    vs = VideoStream(src=0).start()
        vs = cv2.VideoCapture(0)
        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of width pixels
            ##########enter width of frame###########
            width = 800
            #########################################
            ret,frame = vs.read()    #use when reading from videos
            #frame = vs.read()
            frame = imutils.resize(frame, width=width)
            ratio = width/600
            ######
            #cv2.rectangle(frame, (270, 150), (330, 210), (0, 0, 255), 2)
            ######
            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
                arucoDict, parameters=arucoParams)

            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # draw the bounding box of the ArUCo detection
                    cv2.line(frame, topLeft, topRight, (255, 255, 255), 2)
                    cv2.line(frame, topRight, bottomRight, (255, 255, 255), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (255, 255, 255), 2)
                    cv2.line(frame, bottomLeft, topLeft, (255, 255, 255), 2)
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                    ######
                    #cv2.putText(frame, str(cX), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    #cv2.putText(frame, str(cY), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    a1=int(270*ratio)
                    a2=int(330*ratio)
                    a3=int(150*ratio)
                    a4=int(210*ratio)
                    b1=int(200*ratio)
                    b2=int(400*ratio)
                    b3=int(80*ratio)
                    b4=int(280*ratio)
                    c1 = int(((a2-a1)-60)/2)
                    c2 = int(((a4-a3)-60)/2)
                    c3 = int(((b2-b1)-200)/2)
                    c4 = int(((b4-b3)-200)/2)
                    i = int(width/500)
                    if (a1+c1 <= cX <= a2-c1 and a3+c2 <= cY <= a4-c2):
                        cv2.rectangle(frame, (a1+c1, a3+c2), (a2-c1, a4-c2), (0, 255, 0), 5)
                        message = "Bullseye!"
                        cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    elif (b1+c3 <= cX <= b2-c3 and b3+c4 <= cY <= b4-c4):
                        cv2.rectangle(frame, (b1+c3, b3+c4), (b2-c3, b4-c4), (0, 255, 255), 4)
                        if cX < a1+c1:
                            message = "a little to the left"
                            cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                        elif cX > a2-c1:
                            message = "a little to the right"
                            cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                        elif cY < a3+c2:
                            message = "a little down"
                            cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                        elif cX > a4-c2:
                            message = "a little up"
                            cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    if cX < b1+c3:
                            message = "move left"
                            cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    elif cX > b2-c3:
                            message = "move right"
                            cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    elif cY < b3+c4:
                            message = "move down"
                            cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    elif cY > b4-c4:
                            message = "move up"
                            cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    else:
                        message = ""
                        cv2.putText(frame, message, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, i, (0, 0, 255), 3)
                    ######
                    # draw the ArUco marker ID on the frame
                    cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                    # print("[INFO] ArUco marker ID: {}".format(markerID))
                    #Say("Aruco marker number "+str(markerID)+" detected")
                    print("Aruco marker number "+str(markerID)+" detected")
                    #time.sleep(3)
                    if markerID == 1:
                        print("Marker spotted")
                        time.sleep(1)
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()


class ArucoApp(App):
    def __init__(self, address: str, port: int, stream_every_n: int) -> None:
        super().__init__()
        self.address = address
        self.port = port
        self.stream_every_n = stream_every_n

        self.image_decoder = TurboJPEG()
        self.tasks: List[asyncio.Task] = []

    def build(self):
        self.ma = MyAruco()
        return Builder.load_file("res/main.kv")

    def on_exit_btn(self) -> None:
        """Kills the running kivy application."""
        App.get_running_app().stop()

    async def app_func(self):
        async def run_wrapper():
            # we don't actually need to set asyncio as the lib because it is
            # the default, but it doesn't hurt to be explicit
            await self.async_run(async_lib="asyncio")
            for task in self.tasks:
                task.cancel()

        # configure the camera client
        config = ClientConfig(address=self.address, port=self.port)
        client = OakCameraClient(config)

        # Stream camera frames
        self.tasks.append(asyncio.ensure_future(self.stream_camera(client)))

        return await asyncio.gather(run_wrapper(), *self.tasks)

    async def stream_camera(self, client: OakCameraClient) -> None:
        """This task listens to the camera client's stream and populates the tabbed panel with all 4 image streams
        from the oak camera."""
        while self.root is None:
            await asyncio.sleep(0.01)

        response_stream = None
        dim = None  # Will hold rgb image dimensions
        while True:
            # check the state of the service
            state = await client.get_state()

            if state.value not in [
                service_pb2.ServiceState.IDLE,
                service_pb2.ServiceState.RUNNING,
            ]:
                # Cancel existing stream, if it exists
                if response_stream is not None:
                    response_stream.cancel()
                    response_stream = None
                print("Camera service is not streaming or ready to stream")
                await asyncio.sleep(0.1)
                continue

            # Create the stream
            if response_stream is None:
                response_stream = client.stream_frames(every_n=self.stream_every_n)

            try:
                # try/except so app doesn't crash on killed service
                response: oak_pb2.StreamFramesReply = await response_stream.read()
                assert response and response != grpc.aio.EOF, "End of stream"
            except Exception as e:
                print(e)
                response_stream.cancel()
                response_stream = None
                continue

            # get the sync frame
            frame: oak_pb2.OakSyncFrame = response.frame

            reply: oak_pb2.GetCalibrationReply = await client.get_calibration()
            calibration: oak_pb2.OakCalibration = reply.calibration

            # Below is how we get camera matrix and distortion coefficients from calibration data
            dist = np.array(
                calibration.camera_data[2].distortion_coeff
            )  # the 2 in both of these means the right camera
            mtx = np.array(calibration.camera_data[2].intrinsic_matrix).reshape(3, 3)

            # get image and show
            for view_name in ["rgb", "disparity", "left", "right"]:
                # Skip if view_name was not included in frame
                try:
                    # Decode the image and render it in the correct kivy texture
                    img = self.image_decoder.decode(
                        getattr(frame, view_name).image_data
                    )

                    # Set RGB dimensions
                    if view_name == "rgb" and dim is None:
                        width = int(img.shape[1])  # * scale_percent / 100)
                        height = int(img.shape[0])  # * scale_percent / 100)
                        dim = (width, height)
                        print("image dim: ", dim)

                    # Trying to get a depth map:
                    if view_name == "disparity":
                        disp_img = img[:, :, 0]  # because there are too many channels
                        base_line = (
                            2.0
                            * calibration.camera_data[2].extrinsics.spec_translation.x
                        )  # in cm
                        focal_length_pix = calibration.camera_data[2].intrinsic_matrix[
                            0
                        ]  # focal length in pixels
                        depth = np.clip(
                            focal_length_pix
                            * base_line
                            / (100.0 * (disp_img + 1.0 * 10e-8)),
                            None,
                            10.0,
                        )  # clip depths > 10m
                        # Get the dimensions of the image
                        height, width = depth.shape

                        # Calculate the center point of the image
                        center_x = int(width / 2)
                        center_y = int(height / 2)
                        print("center is [{}, {}]".format(center_x, center_y))
                        rescaled_array = cv2.normalize(
                            depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                        )
                        # Define color map
                        colormap = cv2.COLORMAP_JET

                        # Apply the color map to the depth image
                        colored_depth = cv2.applyColorMap(rescaled_array, colormap)
                        #cv2.imshow("depth", colored_depth)
                        cv2.waitKey(10)
                    # # trying to compute the depth above, pretty sure the image part is wrong

                    # New trial testing the rvecs and tvecs outputted:
                    corners, ids, frame_markers = self.ma.detect(img)
                    if corners:
                        rvecs, tvecs = self.ma.pose_estimation(
                            frame_markers, corners, ids, mtx, dist
                        )  # did not need to equal rvecs and t but could use later
                        img = frame_markers

                    # # Scale up images for consistent viewing
                    # if int(img.shape[1]) != dim[0]:  # If img width is different, make it the same
                    #     img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                    texture = Texture.create(
                        size=(img.shape[1], img.shape[0]), icolorfmt="bgr"
                    )
                    texture.flip_vertical()
                    texture.blit_buffer(
                        img.tobytes(),
                        colorfmt="bgr",
                        bufferfmt="ubyte",
                        mipmap_generation=False,
                    )
                    self.root.ids[view_name].texture = texture

                except Exception as e:
                    print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--port", type=int, required=True, help="The camera port.")
    parser.add_argument(
        "--address", type=str, default="localhost", help="The camera address"
    )
    parser.add_argument(
        "--stream-every-n", type=int, default=1, help="Streaming frequency"
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="DICT_ARUCO_ORIGINAL",
        help="type of ArUCo tag to generate such as: DICT_ARUCO_ORIGINAL (default)",
    )  # I think that i should make this not default
    parser.add_argument(
        "-s",
        "--size_marker",
        type=float,
        default=0.0145,
        help="Size of Aruco marker in meters (0.0145 Default)",
    )  # delete default, eventually want to add in a setting in the app if possible
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            ArucoApp(args.address, args.port, args.stream_every_n).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()
