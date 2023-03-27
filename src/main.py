import argparse
import asyncio
import os
from typing import List
import grpc
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig
from turbojpeg import TurboJPEG

import cv2
import face_recognition
import pickle

#set up pickle for facial recognition
currentname = "unknown"
encodingsP = "encodings.pickle"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

os.environ["KIVY_NO_ARGS"] = "1"


from kivy.config import Config  # noreorder # noqa: E402
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", "1280")
Config.set("graphics", "height", "800")
Config.set("graphics", "fullscreen", "false")
Config.set("input", "mouse", "mouse,disable_on_activity")
Config.set("kivy", "keyboard_mode", "systemanddock")
from kivy.app import App  # noqa: E402
from kivy.lang.builder import Builder  # noqa: E402
from kivy.graphics.texture import Texture  # noqa: E402
class CameraApp(App):
    def __init__(self, address: str, port: int, stream_every_n: int) -> None:
        super().__init__()
        self.address = address
        self.port = port
        self.stream_every_n = stream_every_n
        self.image_decoder = TurboJPEG()
        self.tasks: List[asyncio.Task] = []
    def build(self):
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
            # get image and show
            for view_name in ["rgb", "disparity", "left", "right"]:
                # Skip if view_name was not included in frame
                try:
                    # Decode the image and render it in the correct kivy texture
                    img = self.image_decoder.decode(
                        getattr(frame, view_name).image_data
                    )

                    # Demo opencv use - draw a red circle in the center of the image
                    cv2.circle(img,(img.shape[1]//2, img.shape[0]//2), 100, (0,0,255), -1)
                    
                    ###########commands for facial recognition###########
                    boxes = face_recognition.face_locations(img)
                    encodings = face_recognition.face_encodings(img, boxes)
                    names = []
                    
                    for encoding in encodings:
                        matches = face_recognition.compare_faces(data["encodings"], encoding)
                        name = "Unknown" 
                        if True in matches:
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}
                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1
                            name = max(counts, key=counts.get)
                            if currentname != name:
                                currentname = name
                        names.append(name)
                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
                        y = top - 5 if top - 5 > 5 else top + 5
                        cv2.rectangle(frame, (left, top-20), (int((left+right)/2), top), (0, 255, 255), -1)
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)

                     ###########################################

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
    parser.add_argument("--port", type=int, default=50050, help="The camera port.") #required=True instead of default=0
    parser.add_argument(
        "--address", type=str, default="localhost", help="The camera address"
    )
    parser.add_argument(
        "--stream-every-n", type=int, default=0, help="Streaming frequency"
    )
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            CameraApp(args.address, args.port, args.stream_every_n).app_func()
        )
    except asyncio.CancelledError:
        pass
    loop.close()
