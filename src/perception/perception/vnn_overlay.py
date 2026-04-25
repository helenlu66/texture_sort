import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VnnOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("vnn_overlay")
        self.bridge = CvBridge()

        self.declare_parameter("image_topic", "/wrist_camera/color/image_rect_color/compressed")
        self.declare_parameter("output_topic", "/wrist_camera/color/vnn_detections")
        self.declare_parameter("detector_labels", ["pan", "tomatoe", "eggplant", "red basket", "leek"])
        self.declare_parameter("detection_threshold", 0.15)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        labels = list(self.get_parameter("detector_labels").value)
        self.text_labels = [labels]
        self.detection_threshold = float(self.get_parameter("detection_threshold").value)

        self.pub = self.create_publisher(Image, output_topic, 10)
        self.create_subscription(Image, image_topic, self._callback, 10)

        self.processor = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        self._load_model()

        self.get_logger().info(f"vnn_overlay ready on {image_topic} → {output_topic}")

    def _load_model(self) -> None:
        try:
            import importlib
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = transformers.Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
            self.model = (
                transformers.Owlv2ForObjectDetection.from_pretrained(
                    "google/owlv2-base-patch16-ensemble"
                )
                .to(self.device)
                .eval()
            )
            self.get_logger().info(f"OWLv2 loaded on {self.device}")
        except Exception as exc:
            self.get_logger().warn(f"OWLv2 unavailable ({exc}); will publish passthrough images")

    def _callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Decode failed: {exc}")
            return

        if self.model is not None and self.processor is not None:
            try:
                from PIL import Image as PILImage
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(rgb)
                inputs = self.processor(
                    text=self.text_labels, images=img_pil, return_tensors="pt"
                ).to(self.device)
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                target_sizes = self.torch.tensor([(img_pil.height, img_pil.width)]).to(self.device)
                result = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.detection_threshold,
                    text_labels=self.text_labels,
                )[0]
                self._draw(bgr, result["boxes"], result["scores"], result["text_labels"])
            except Exception as exc:
                self.get_logger().warn(f"OWLv2 inference failed: {exc}")

        out = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        self.pub.publish(out)

    def _draw(self, bgr: np.ndarray, boxes, scores, labels) -> None:
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(bgr, (cx, cy), 4, (0, 0, 255), -1)
            text = f"{label} {float(score):.2f} ({cx},{cy})"
            cv2.putText(bgr, text, (x0, max(y0 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VnnOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
