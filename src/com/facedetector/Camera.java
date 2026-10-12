package com.facedetector;



/** Camera helpers, kept separate from Activity lifecycle and frame processing. */
@SuppressWarnings("deprecation")
public final class Camera {
    private Camera() {}
    public static final class CameraController {
        private CameraController() {}
        public static int chooseCamera(boolean front) {
            android.hardware.Camera.CameraInfo info = new android.hardware.Camera.CameraInfo();
            for (int i = 0; i < android.hardware.Camera.getNumberOfCameras(); i++) {
                android.hardware.Camera.getCameraInfo(i, info);
                if (info.facing == (front ? android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT : android.hardware.Camera.CameraInfo.CAMERA_FACING_BACK)) return i;
            }
            return -1;
        }
    }
    public static final class CameraOrientation {
        private CameraOrientation() {}
        public static int previewRotation(android.hardware.Camera.CameraInfo info, int displayRotation) {
            int degrees = displayRotation == 1 ? 90 : displayRotation == 2 ? 180 : displayRotation == 3 ? 270 : 0;
            if (info.facing == android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT) {
                int rotation = (info.orientation + degrees) % 360;
                return (360 - rotation) % 360;
            }
            return (info.orientation - degrees + 360) % 360;
        }
    }
}
