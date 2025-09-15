# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# import cv2  

# class CourtLineDetector:
#     def __init__(self,model_path):
#         self.model = models.resnet50(pretrained=False)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
#         self.model.load_state_dict(torch.load(model_path, map_location ='cpu'))

#         self.transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def predict(self,image):

#         img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_tensor = self.transforms(img_rgb).unsqueeze(0)  # Add batch dimension


#         with torch.no_grad():
#             outputs = self.model(image_tensor)

#         keypoints = outputs.squeeze().cpu().numpy()
#         original_h, original_w = img_rgb.shape[:2]

#         keypoints[::2] = keypoints[::2] * original_w/224.0
#         keypoints[1::2] = keypoints[1::2] * original_h/224.0

#         return keypoints


#     def draw_keypoints(self,image,keypoints):
#         for i in range(0,len(keypoints),2): 
#             x = int(keypoints[i])
#             y = int(keypoints[i + 1])
            

#             cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#             cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
#         return image
    
#     def draw_keypoints_on_video(self,video_frames,keypoints):
#         output_video_frames = []
#         for frame in video_frames:
#             frame = self.draw_keypoints(frame,keypoints)
#             output_video_frames.append(frame)

#         return output_video_frames


import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path, use_letterbox=False, input_size=224):
        """
        use_letterbox=False assumes training used a plain Resize((224,224))
        If you trained with letterbox-padding, set use_letterbox=True.
        """
        self.input_size = input_size
        self.use_letterbox = use_letterbox

        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()  # important for inference

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ---------- (A) Optional letterbox helpers ----------
    @staticmethod
    def _letterbox(img, new_shape=224, color=(114,114,114)):
        """YOLO-style letterbox to square while keeping aspect."""
        h, w = img.shape[:2]
        r = float(new_shape) / max(h, w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw /= 2; dh /= 2

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return out, r, (left, top)

    @staticmethod
    def _unletterbox_points(kps_xy, r, pad, orig_w, orig_h):
        """Undo letterbox mapping to original image coordinates."""
        kps_xy = kps_xy.copy()
        kps_xy[:, 0] = (kps_xy[:, 0] - pad[0]) / r
        kps_xy[:, 1] = (kps_xy[:, 1] - pad[1]) / r
        kps_xy[:, 0] = np.clip(kps_xy[:, 0], 0, orig_w - 1)
        kps_xy[:, 1] = np.clip(kps_xy[:, 1], 0, orig_h - 1)
        return kps_xy

    # ---------- (B) Core single-image predict ----------
    def predict(self, image_bgr):
        """
        Returns keypoints in original image pixel coordinates (shape: (28,))
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        if self.use_letterbox:
            boxed, r, pad = self._letterbox(img_rgb, new_shape=self.input_size)
            tensor = self.transforms(boxed).unsqueeze(0)
        else:
            tensor = self.transforms(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)  # (1, 28)

        kps = outputs.squeeze().cpu().numpy()  # (28,)
        kps_xy = kps.reshape(-1, 2)  # (14,2) in model input space

        if self.use_letterbox:
            # undo letterbox to original coords
            kps_xy = self._unletterbox_points(kps_xy, r, pad, orig_w, orig_h)
        else:
            # plain-resize mapping back to original size
            sx = float(orig_w) / float(self.input_size)
            sy = float(orig_h) / float(self.input_size)
            kps_xy[:, 0] *= sx
            kps_xy[:, 1] *= sy

        return kps_xy.reshape(-1)  # (28,)

    # ---------- (C) Per-frame prediction with optional smoothing ----------
    def predict_frames(self, video_frames, smooth_alpha=0.0):
        """
        Predicts keypoints for each frame.
        If smooth_alpha > 0, applies EMA temporal smoothing.
        Returns: list of (28,) float arrays, len == len(video_frames)
        """
        preds = []
        prev = None
        for frame in video_frames:
            kps = self.predict(frame)  # (28,)
            if smooth_alpha > 0 and prev is not None:
                kps = smooth_alpha * kps + (1.0 - smooth_alpha) * prev
            preds.append(kps)
            prev = kps
        return preds

    # ---------- (D) Drawing ----------
    @staticmethod
    def draw_keypoints(image, keypoints):
        # keypoints: flat (28,) -> x,y pairs
        for i in range(0, len(keypoints), 2):
            x = int(round(keypoints[i]))
            y = int(round(keypoints[i + 1]))
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints_seq):
        """
        keypoints_seq: list of (28,) arrays aligned with video_frames
        """
        out = []
        for frame, kps in zip(video_frames, keypoints_seq):
            frame = self.draw_keypoints(frame, kps)
            out.append(frame)
        return out

