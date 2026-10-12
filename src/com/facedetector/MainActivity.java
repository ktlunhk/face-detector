package com.facedetector;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.widget.ImageView;
import java.io.IOException;
import android.Manifest;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Button;
import android.view.Gravity;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.Drawable;
import android.content.res.ColorStateList;
import android.content.Context;
import android.graphics.RectF;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;

@SuppressWarnings("deprecation")
public class MainActivity extends Activity implements SurfaceHolder.Callback, Camera.PreviewCallback {
    private static final int PICK_IMAGE = 202;
    private final boolean useYuNet = true;
    private FaceDetectorYN yuNet;
    private Net faceMeshNet;
    private volatile String meshDiagnostic = "OFF";
    private final Object meshLock = new Object();
    private volatile String yuNetDiagnostics = "YuNet not yet run";
    private volatile float yuNetConfidence = 0.70f;
    private volatile float yuNetNms = 0.30f;
    private volatile int yuNetTopK = 500;
    private final Object yuNetLock = new Object();
    private Button engineButton;
    private boolean imageMode = false;
    private ImageFaceView imageView;
    private Bitmap selectedBitmap;
    private Button loadButton, cameraButton, meshButton;
    private volatile boolean meshEnabled = false;
    private int imageGeneration = 0;
    private Camera camera;
    private SurfaceView surface;
    private Rendering.Overlay overlay;
    private TextView status;
    private CascadeClassifier classifier;
    private int cameraId = -1;
    private int width = 640, height = 480, rotation = 90;
    private boolean busy = false;
    private long lastScan = 0;
    private boolean opencvReady = false;
    private LinearLayout root;
    private FrameLayout previewArea;
    private LinearLayout settingsPanel;
    private float scaleFactor = 1.10f, smoothing = 0.50f;
    private int minNeighbors = 5, minFace = 70, interval = 150, holdMs = 400;
    private RectF[] displayed = new RectF[0];
    private long lastDetected = 0;
    private SharedPreferences prefs;
    private int frameRotation = 90;
    private boolean frontFacing = false;
    private void saveSettings() {
        Settings.save(prefs, scaleFactor, minNeighbors, minFace, interval, holdMs,
                smoothing, yuNetConfidence, yuNetNms, yuNetTopK);
    }
    private Ui.Style uiStyle;
    private GradientDrawable rounded(int color, int stroke, int radius) { return uiStyle.rounded(color, stroke, radius); }
    private Drawable icon(int type, int color) { return uiStyle.icon(type, color); }
    private void setIcon(TextView view, int type, int color) { uiStyle.setIcon(view, type, color); }
    private void styleAction(Button button, boolean selected) { uiStyle.styleAction(button, selected); }
    private LinearLayout.LayoutParams buttonParams() { return uiStyle.buttonParams(); }
    private void makeSlider(LinearLayout panel, final String name, int initial, final int min, final int step, int max, final int kind) {
        final TextView label = new TextView(this);
        label.setTextColor(0xFFE2EAF4); label.setPadding(dp(5),dp(3),dp(5),0);
        setIcon(label, kind == 6 ? 4 : kind == 7 ? 5 : kind == 8 ? 6 : kind == 3 ? 7 : kind == 4 ? 8 : 9, 0xFFB7D7E6);
        panel.addView(label);
        SeekBar bar = new SeekBar(this);
        bar.setMax((max-min)/step);
        if (android.os.Build.VERSION.SDK_INT >= 21) {
            bar.setProgressTintList(ColorStateList.valueOf(0xFF22D3EE));
            bar.setThumbTintList(ColorStateList.valueOf(0xFF22D3EE));
            bar.setProgressBackgroundTintList(ColorStateList.valueOf(0xFF405267));
        }
        panel.addView(bar, new LinearLayout.LayoutParams(-1, dp(29)));
        bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b,int progress,boolean user) {
                int value = min + progress*step;
                if (kind==0) {scaleFactor=value/100f;label.setText(name+": "+scaleFactor);}
                if (kind==1) {minNeighbors=value;label.setText(name+": "+value+(value<4?" (possible false positives)":""));}
                if (kind==2) {minFace=value;label.setText(name+": "+value+" px");}
                if (kind==3) {interval=value;label.setText(name+": "+value+" ms");}
                if (kind==4) {holdMs=value;label.setText(name+": "+value+" ms");}
                if (kind==5) {smoothing=value/100f;label.setText(name+": "+value+"%");}
                if (kind==6) {yuNetConfidence=value/100f;label.setText(name+": "+value+"%");}
                if (kind==7) {yuNetNms=value/100f;label.setText(name+": "+value+"%");}
                if (kind==8) {yuNetTopK=value;label.setText(name+": "+value);}
                if (user && kind >= 6) {
                    synchronized (yuNetLock) {
                        if (yuNet != null) {
                            yuNet.setScoreThreshold(yuNetConfidence);
                            yuNet.setNMSThreshold(yuNetNms);
                            yuNet.setTopK(yuNetTopK);
                        }
                    }
                }
                if (user) {
                    saveSettings();
                    if (imageMode && selectedBitmap != null) detectSelectedImage();
                }
            }
            public void onStartTrackingTouch(SeekBar b) {}
            public void onStopTrackingTouch(SeekBar b) {}
        });
        // Initialize after attaching listener, so the label is never blank.
        bar.setProgress(Math.max(0, Math.min(bar.getMax(), (initial-min)/step)));
        // Some Android versions skip the callback when progress remains zero.
        int current = min + bar.getProgress() * step;
        if (kind == 0) label.setText(name + ": " + (current / 100f));
        else if (kind == 1) label.setText(name + ": " + current + (current < 4 ? " (possible false positives)" : ""));
        else if (kind == 2) label.setText(name + ": " + current + " px");
        else if (kind == 3 || kind == 4) label.setText(name + ": " + current + " ms");
        else if (kind == 8) label.setText(name + ": " + current);
        else label.setText(name + ": " + current + "%");
        label.setTextSize(14);
        label.setMinHeight(dp(21));
    }
    private void createSettings() {
        settingsPanel = new LinearLayout(this);
        settingsPanel.setOrientation(LinearLayout.VERTICAL);
        settingsPanel.setPadding(dp(12), dp(6), dp(12), dp(6));
        settingsPanel.setBackground(rounded(0xFF142236, 0xFF2A4159, 20));
        TextView heading = new TextView(this);
        heading.setText("Face detection settings (live)");
        setIcon(heading, 10, 0xFFE2EAF4);
        heading.setTextSize(16);
        heading.setTextColor(Color.WHITE);
        heading.setPadding(dp(5), dp(2), dp(5), dp(3));
        settingsPanel.addView(heading);
        makeSlider(settingsPanel,"Confidence threshold",Math.round(yuNetConfidence*100),10,5,95,6);
        makeSlider(settingsPanel,"NMS threshold",Math.round(yuNetNms*100),5,5,90,7);
        makeSlider(settingsPanel,"Max detections",yuNetTopK,100,100,2000,8);
        makeSlider(settingsPanel,"Interval",interval,50,50,1000,3);
        makeSlider(settingsPanel,"Hold time",holdMs,0,50,2000,4);
        makeSlider(settingsPanel,"Smoothing",Math.round(smoothing*100),0,5,95,5);
        android.widget.ScrollView scroll = new android.widget.ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setVerticalScrollBarEnabled(true);
        scroll.setSmoothScrollingEnabled(true);
        scroll.addView(settingsPanel);
        scroll.setBackgroundColor(0xFF091420);
        scroll.setPadding(dp(10), dp(3), dp(10), dp(4));
        scroll.setClipToPadding(false);
        // Dedicated lower region: never overlays the image or action buttons.
        // Scroll only the settings. The preview and action buttons remain fixed.
        LinearLayout.LayoutParams settingsLp = new LinearLayout.LayoutParams(-1, 0, 1.55f);
        root.addView(scroll, settingsLp);
    }
    private int dp(int value) {
        return (int)(value * getResources().getDisplayMetrics().density + 0.5f);
    }
    private void updatePreviewLayout() {
        if (surface == null || overlay == null || previewArea == null) return;
        int sw = previewArea.getWidth();
        int sh = previewArea.getHeight();
        if (sw <= 0 || sh <= 0) return;
        int iw = (frameRotation == 90 || frameRotation == 270) ? height : width;
        int ih = (frameRotation == 90 || frameRotation == 270) ? width : height;
        if (iw <= 0 || ih <= 0) return;
        float factor = Math.min((float) sw / iw, (float) sh / ih);
        int vw = Math.max(1, Math.round(iw * factor));
        int vh = Math.max(1, Math.round(ih * factor));
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(vw, vh, Gravity.CENTER);
        surface.setLayoutParams(lp);
        overlay.setLayoutParams(new FrameLayout.LayoutParams(vw, vh, Gravity.CENTER));
        // ImageFaceView fits the bitmap inside its bounds using uniform scaling.
        imageView.setLayoutParams(new FrameLayout.LayoutParams(-1, -1));
    }
    private RectF[] stabilize(RectF[] next) {
        long now=System.currentTimeMillis();
        if(next.length==0) {
            if(now-lastDetected<=holdMs) return displayed;
            displayed=new RectF[0];return displayed;
        }
        lastDetected=now;
        if(next.length==1 && displayed.length==1) {
            RectF a=displayed[0], b=next[0];
            float cx=(a.centerX()-b.centerX()),cy=(a.centerY()-b.centerY());
            if(cx*cx+cy*cy<0.09f) {
                float k=1f-smoothing;
                next[0]=new RectF(a.left*(1-k)+b.left*k,a.top*(1-k)+b.top*k,
                    a.right*(1-k)+b.right*k,a.bottom*(1-k)+b.bottom*k);
            }
        }
        displayed=next;return displayed;
    }

    public void onCreate(Bundle b) {
        super.onCreate(b);
        uiStyle = new Ui.Style(this);
        prefs=getSharedPreferences("face_settings",MODE_PRIVATE);
        scaleFactor=prefs.getFloat("scale",1.10f);minNeighbors=prefs.getInt("neighbors",5);
        minFace=prefs.getInt("face",70);interval=prefs.getInt("interval",150);
        holdMs=prefs.getInt("hold",400);smoothing=prefs.getFloat("smooth",0.5f);
        yuNetConfidence=prefs.getFloat("yunet_conf",0.70f);
        yuNetNms=prefs.getFloat("yunet_nms",0.30f);
        yuNetTopK=prefs.getInt("yunet_topk",500);
        meshEnabled=prefs.getBoolean("mesh_enabled",false);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF091420);
        status = new TextView(this);
        status.setTextColor(Color.WHITE);
        status.setTextSize(14);
        status.setBackgroundColor(0xFF202020);
        status.setPadding(dp(12), dp(8), dp(12), dp(8));
        status.setText("Starting OpenCV...");
        // Keep diagnostics available for error handling without occupying the top of the screen.
        status.setVisibility(View.GONE);

        TextView appTitle = new TextView(this);
        appTitle.setText("Face Detector");
        appTitle.setTextColor(Color.WHITE);
        appTitle.setTextSize(23);
        appTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        appTitle.setGravity(Gravity.CENTER_VERTICAL);
        appTitle.setPadding(dp(18), dp(9), dp(16), dp(7));
        root.addView(appTitle, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(dp(7), 0, dp(7), 0);
        loadButton = new Button(this);
        loadButton.setText("Load image");
        styleAction(loadButton, false);
        setIcon(loadButton, 0, 0xFFE8F2FF);
        actions.addView(loadButton, buttonParams());
        cameraButton = new Button(this);
        cameraButton.setText("Camera");
        styleAction(cameraButton, false);
        setIcon(cameraButton, 1, 0xFFE8F2FF);
        actions.addView(cameraButton, buttonParams());
        root.addView(actions, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout modes = new LinearLayout(this);
        modes.setOrientation(LinearLayout.HORIZONTAL);
        modes.setPadding(dp(7), 0, dp(7), dp(2));
        Button switchCameraButton = new Button(this);
        switchCameraButton.setText("Switch camera");
        styleAction(switchCameraButton, false);
        setIcon(switchCameraButton, 2, 0xFFE8F2FF);
        modes.addView(switchCameraButton, buttonParams());
        switchCameraButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { switchCamera(); }
        });
        meshButton = new Button(this);
        meshButton.setText(meshEnabled ? "Face mesh  ON" : "Face mesh  OFF");
        styleAction(meshButton, meshEnabled);
        setIcon(meshButton, 3, meshEnabled ? 0xFF06131E : 0xFFE8F2FF);
        modes.addView(meshButton, buttonParams());
        root.addView(modes, new LinearLayout.LayoutParams(-1, -2));
        meshButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                meshEnabled = !meshEnabled;
                prefs.edit().putBoolean("mesh_enabled", meshEnabled).apply();
                meshButton.setText(meshEnabled ? "Face mesh  ON" : "Face mesh  OFF");
                styleAction(meshButton, meshEnabled);
        setIcon(meshButton, 3, meshEnabled ? 0xFF06131E : 0xFFE8F2FF);
                if (overlay != null) {
                    overlay.meshEnabled = meshEnabled;
                    overlay.invalidate();
                }
                if (imageMode && selectedBitmap != null) detectSelectedImage();
            }
        });
        previewArea = new FrameLayout(this);
        previewArea.setBackground(rounded(0xFF020B12, 0xFF2A4159, 16));
        previewArea.setClipToOutline(true);
        LinearLayout.LayoutParams previewLp = new LinearLayout.LayoutParams(-1, 0, 0.85f);
        previewLp.setMargins(dp(10), dp(4), dp(10), dp(3));
        root.addView(previewArea, previewLp);
        surface = new SurfaceView(this);
        previewArea.addView(surface);
        overlay = new Rendering.Overlay(this);
        overlay.meshEnabled = meshEnabled;
        previewArea.addView(overlay);
        imageView = new ImageFaceView(this);
        imageView.setVisibility(View.GONE);
        previewArea.addView(imageView, new FrameLayout.LayoutParams(-1, -1));
        previewArea.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            public void onLayoutChange(View v, int left, int top, int right, int bottom,
                int oldLeft, int oldTop, int oldRight, int oldBottom) {
                if (right-left != oldRight-oldLeft || bottom-top != oldBottom-oldTop) updatePreviewLayout();
            }
        });
        loadButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { pickImage(); }
        });
        cameraButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { showCamera(); }
        });
        createSettings();
        setContentView(root);
        surface.getHolder().addCallback(this);

        try {
            System.loadLibrary("opencv_java4");
            opencvReady = true;
        } catch (Throwable e) {
            status.setText("OpenCV load failed. Check native libraries");
        }
        if (opencvReady) {
            try { ensureBundledYuNetModel(); }
            catch (Throwable e) { status.setText("Classifier error: " + e.getMessage()); }
        }
        if (android.os.Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 101);
        }
    }

    // Copy the bundled ONNX model into internal storage because FaceDetectorYN
    // requires a filesystem path rather than an Android asset stream.
    private void ensureBundledYuNetModel() throws IOException {
        File target = new File(getFilesDir(), "yunet.onnx");
        InputStream input = getAssets().open("face_detection_yunet_2026may.onnx");
        File temp = new File(getFilesDir(), "yunet_bundled.tmp");
        try {
            FileOutputStream output = new FileOutputStream(temp);
            try {
                byte[] buffer = new byte[16384];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            } finally { output.close(); }
        } finally { input.close(); }
        if (temp.length() < 100000) { temp.delete(); throw new IOException("Bundled model is incomplete"); }
        if (target.exists() && target.length() == temp.length()) { temp.delete(); return; }
        synchronized (yuNetLock) {
            if (target.exists() && !target.delete()) throw new IOException("Cannot replace model");
            if (!temp.renameTo(target)) throw new IOException("Cannot install bundled model");
            yuNet = null;
        }
    }

    private void loadClassifier() throws Exception {
        File f = new File(getFilesDir(), "haarcascade_frontalface_default.xml");
        InputStream in = getAssets().open("haarcascade_frontalface_default.xml");
        FileOutputStream out = new FileOutputStream(f);
        try {
            byte[] buf = new byte[8192]; int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        } finally { try { in.close(); } catch (Exception ignored) {} try { out.close(); } catch (Exception ignored) {} }
        classifier = new CascadeClassifier(f.getAbsolutePath());
        if (classifier.empty()) throw new Exception("Cannot load Haar XML");
    }

    public void onRequestPermissionsResult(int req, String[] p, int[] grants) {
        super.onRequestPermissionsResult(req, p, grants);
        if (req == 101 && grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED) startCamera();
        else status.setText("Camera permission required");
    }

    private void switchCamera() {
        if (imageMode) return;
        int front = com.facedetector.Camera.CameraController.chooseCamera(true), back = com.facedetector.Camera.CameraController.chooseCamera(false);
        int next = cameraId == back ? front : back;
        if (next < 0) next = back;
        stopCamera(); cameraId = next; startCamera();
    }

    private void startCamera() {
        if (imageMode || !opencvReady) return;
        if (camera != null || !surface.getHolder().getSurface().isValid()) return;
        if (android.os.Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        try {
            if (cameraId < 0) cameraId = com.facedetector.Camera.CameraController.chooseCamera(false);
            if (cameraId < 0) throw new Exception("No camera available");
            camera = Camera.open(cameraId);
            Camera.Parameters params = camera.getParameters();
            List<Camera.Size> sizes = params.getSupportedPreviewSizes();
            Camera.Size chosen = sizes.get(0);
            int best = Integer.MAX_VALUE;
            for (Camera.Size s : sizes) {
                int score = Math.abs(s.width - 640) + Math.abs(s.height - 480);
                if (score < best) { best = score; chosen = s; }
            }
            width = chosen.width; height = chosen.height;
            params.setPreviewSize(width, height);
            params.setPreviewFormat(android.graphics.ImageFormat.NV21);
            camera.setParameters(params);
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(cameraId, info);
            int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
            rotation = com.facedetector.Camera.CameraOrientation.previewRotation(info, displayRotation);
            camera.setDisplayOrientation(rotation);
            frontFacing = info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT;
            overlay.front = frontFacing;
            frameRotation=rotation;
            updatePreviewLayout();
            camera.setPreviewDisplay(surface.getHolder());
            camera.setPreviewCallback(this);
            camera.startPreview();
            status.setText("Faces: 0   |   Tap to switch camera");
        } catch (Throwable e) { stopCamera(); status.setText("Camera error: " + e.getMessage()); }
    }

    public void onPreviewFrame(byte[] data, Camera c) {
        if (imageMode || busy || System.currentTimeMillis() - lastScan < interval) return;
        busy = true; lastScan = System.currentTimeMillis();
        Mat yuv = null, gray = null, upright = null, small = null; MatOfRect faces = null;
        try {
            yuv = new Mat(height + height / 2, width, CvType.CV_8UC1);
            yuv.put(0, 0, data);
            gray = new Mat();
            Imgproc.cvtColor(yuv, gray, Imgproc.COLOR_YUV2GRAY_NV21);
            upright = new Mat();
            if (rotation == 90) Core.rotate(gray, upright, Core.ROTATE_90_CLOCKWISE);
            else if (rotation == 180) Core.rotate(gray, upright, Core.ROTATE_180);
            else if (rotation == 270) Core.rotate(gray, upright, Core.ROTATE_90_COUNTERCLOCKWISE);
            else gray.copyTo(upright);
            small = new Mat();
            double scale = Math.min(1.0, 480.0 / upright.cols());
            Imgproc.resize(upright, small, new Size(), scale, scale, Imgproc.INTER_LINEAR);
            if (!useYuNet) Imgproc.equalizeHist(small, small);
            double[][] found;
            if (useYuNet) {
                Mat bgr = new Mat();
                try {
                    Imgproc.cvtColor(small, bgr, Imgproc.COLOR_GRAY2BGR);
                    found = detectYuNet(bgr);
                } finally { bgr.release(); }
            } else {
                faces = new MatOfRect();
                synchronized (classifier) { classifier.detectMultiScale(small, faces, scaleFactor, minNeighbors, 0, new Size(Math.max(1,minFace*scale), Math.max(1,minFace*scale)), new Size()); }
                found = readFaceRects(faces);
            }
            final StringBuilder cameraDebug = new StringBuilder();
            cameraDebug.append("Detect ").append(small.cols()).append("x").append(small.rows())
                .append(" | upright ").append(upright.cols()).append("x").append(upright.rows());
            for (int i = 0; i < Math.min(found.length, 3); i++) {
                double[] r = found[i];
                cameraDebug.append(String.format(java.util.Locale.US,
                    "\n#%d raw [%.0f,%.0f,%.0f,%.0f]", i + 1, r[0], r[1], r[2], r[3]));
            }
            if (useYuNet && latestYuNetRows.length > 0) {
                double[] r = latestYuNetRows[0];
                cameraDebug.append(String.format(java.util.Locale.US,
                    "\nEyes (%.0f,%.0f) (%.0f,%.0f) nose (%.0f,%.0f) mouth (%.0f,%.0f) (%.0f,%.0f)",
                    r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]));
            }
            cameraDebug.append(" | front=").append(frontFacing).append(" rotation=").append(rotation)
                .append(" | Y flip=").append(frontFacing);
            final RectF[] rects = new RectF[found.length];
            for (int i = 0; i < found.length; i++) {
                double[] r = found[i];
                // Front preview is vertically inverted relative to the rotated NV21
                // detector matrix on this device. Keep X unchanged (v10.0 fix),
                // and invert Y in detector coordinates before normalization.
                float left = (float)(r[0] / small.cols());
                float right = (float)((r[0] + r[2]) / small.cols());
                float top = frontFacing
                        ? (float)((small.rows() - r[1] - r[3]) / small.rows())
                        : (float)(r[1] / small.rows());
                float bottom = frontFacing
                        ? (float)((small.rows() - r[1]) / small.rows())
                        : (float)((r[1] + r[3]) / small.rows());
                rects[i] = new RectF(left, top, right, bottom);
            }
            // The detector frame is vertically inverted relative to the front
            // preview. Flip the model crop before inference, then map its
            // landmarks back into detector coordinates and preview coordinates.
            final float[][] meshPoints = meshEnabled && useYuNet
                    ? estimateMeshes(small, found, frontFacing) : new float[0][];
            runOnUiThread(new Runnable() {
                public void run() {
                    overlay.meshPoints = meshPoints;
                    overlay.rects = stabilize(rects);
                    overlay.meshEnabled = meshEnabled;
                    overlay.invalidate();
                    status.setText("YuNet faces: " + overlay.rects.length + " | Tap to switch camera"
                        + (meshEnabled ? " | Mesh: " + meshDiagnostic : ""));
                }
            });
        } catch (Throwable e) {
            final String msg = e.getMessage();
            runOnUiThread(new Runnable() { public void run() { status.setText("Detection error: " + msg); } });
        } finally {
            if (faces != null) faces.release(); if (small != null) small.release();
            if (upright != null) upright.release(); if (gray != null) gray.release(); if (yuv != null) yuv.release();
            busy = false;
        }
    }

    private void stopCamera() {
        if (camera != null) {
            try { camera.setPreviewCallback(null); camera.stopPreview(); camera.release(); } catch (Exception ignored) {}
            camera = null;
        }
        overlay.rects = new RectF[0]; displayed=new RectF[0]; overlay.invalidate();
    }
    public void surfaceCreated(SurfaceHolder h) { startCamera(); }
    public void surfaceChanged(SurfaceHolder h, int format, int w, int ht) { if (camera == null) startCamera(); }
    public void surfaceDestroyed(SurfaceHolder h) { stopCamera(); }
    protected void onPause() { stopCamera(); super.onPause(); }
    protected void onResume() { super.onResume(); if (!imageMode && surface != null) startCamera(); }
    protected void onDestroy() {
        stopCamera();
        imageGeneration++;
        if (selectedBitmap != null) { selectedBitmap.recycle(); selectedBitmap = null; }
        super.onDestroy();
    }


    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        try { startActivityForResult(intent, PICK_IMAGE); }
        catch (Exception e) { status.setText("Cannot open image picker: " + e.getMessage()); }
    }

    private void showCamera() {
        imageGeneration++;
        imageMode = false;
        imageView.setVisibility(View.GONE);
        surface.setVisibility(View.VISIBLE);
        overlay.setVisibility(View.VISIBLE);
        if (selectedBitmap != null) { selectedBitmap.recycle(); selectedBitmap = null; }
        status.setText("Starting camera...");
        startCamera();
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_IMAGE || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        Bitmap bitmap = null;
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            InputStream input = getContentResolver().openInputStream(uri);
            try { BitmapFactory.decodeStream(input, null, opts); } finally { if (input != null) input.close(); }
            int sample = 1;
            while (opts.outWidth / sample > 1600 || opts.outHeight / sample > 1600) sample *= 2;
            opts = new BitmapFactory.Options();
            opts.inSampleSize = sample;
            input = getContentResolver().openInputStream(uri);
            try { bitmap = BitmapFactory.decodeStream(input, null, opts); }
            finally { if (input != null) input.close(); }
            if (bitmap == null) throw new IOException("Unsupported image");
            int orientation = ExifInterface.ORIENTATION_NORMAL;
            if (android.os.Build.VERSION.SDK_INT >= 24) {
                input = getContentResolver().openInputStream(uri);
                try { orientation = new ExifInterface(input).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL); }
                finally { if (input != null) input.close(); }
            }
            Matrix transform = new Matrix();
            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) transform.postRotate(90);
            else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) transform.postRotate(180);
            else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) transform.postRotate(270);
            else if (orientation == ExifInterface.ORIENTATION_FLIP_HORIZONTAL) transform.postScale(-1, 1);
            else if (orientation == ExifInterface.ORIENTATION_TRANSPOSE) { transform.postRotate(90); transform.postScale(-1, 1); }
            else if (orientation == ExifInterface.ORIENTATION_TRANSVERSE) { transform.postRotate(270); transform.postScale(-1, 1); }
            else if (orientation == ExifInterface.ORIENTATION_FLIP_VERTICAL) transform.postScale(1, -1);
            if (!transform.isIdentity()) {
                Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), transform, true);
                if (rotated != bitmap) { bitmap.recycle(); bitmap = rotated; }
            }
            imageGeneration++;
            stopCamera();
            imageMode = true;
            surface.setVisibility(View.GONE);
            overlay.setVisibility(View.GONE);
            imageView.setVisibility(View.VISIBLE);
            if (selectedBitmap != null) selectedBitmap.recycle();
            selectedBitmap = bitmap;
            imageView.setImage(bitmap);
            status.setText("Detecting image...");
            detectSelectedImage();
        } catch (Throwable e) {
            if (bitmap != null && bitmap != selectedBitmap) bitmap.recycle();
            status.setText("Image load error: " + e.getMessage());
        }
    }

    // YuNet returns a CV_32F matrix with 15 columns per detection:
    // x, y, width, height, 5 landmark pairs, confidence.
    private volatile double[][] latestYuNetRows = new double[0][];
    private double[][] detectYuNet(Mat bgr) throws Exception {
        synchronized (yuNetLock) {
            File model = new File(getFilesDir(), "yunet.onnx");
            if (!model.exists()) ensureBundledYuNetModel();
            if (yuNet == null) yuNet = FaceDetectorYN.create(model.getAbsolutePath(), "",
                new Size(bgr.cols(), bgr.rows()), yuNetConfidence, yuNetNms, yuNetTopK);
            if (yuNet == null) throw new IOException("Cannot initialize YuNet");
            yuNet.setInputSize(new Size(bgr.cols(), bgr.rows()));
            Mat results = new Mat();
            try {
                int detectionResult = yuNet.detect(bgr, results);
                yuNetDiagnostics = "YuNet model=" + model.length() + " bytes"
                    + " | input=" + bgr.cols() + "x" + bgr.rows()
                    + " ch=" + bgr.channels()
                    + " | return=" + detectionResult
                    + " | output=" + results.rows() + "x" + results.cols()
                    + " type=" + results.type()
                    + " | threshold=" + yuNetConfidence
                    + " | NMS=" + yuNetNms + " | topK=" + yuNetTopK;
                java.util.ArrayList<double[]> found = new java.util.ArrayList<double[]>();
                if (results.rows() > 0 && results.cols() >= 15) {
                    double[] confidence = results.get(0, 14);
                    if (confidence != null && confidence.length > 0)
                        yuNetDiagnostics += String.format(java.util.Locale.US,
                            " | first confidence=%.3f", confidence[0]);
                }
                java.util.ArrayList<double[]> allRows = new java.util.ArrayList<double[]>();
                for (int i = 0; i < results.rows(); i++) {
                    double[] row = new double[15];
                    for (int j = 0; j < 15; j++) {
                        double[] cell = results.get(i, j);
                        if (cell == null || cell.length == 0) throw new IOException("Invalid YuNet result");
                        row[j] = cell[0];
                    }
                    if (row[2] > 0 && row[3] > 0) {
                        found.add(new double[] {row[0],row[1],row[2],row[3]});
                        allRows.add(row);
                    }
                }
                latestYuNetRows = allRows.toArray(new double[allRows.size()][]);
                return found.toArray(new double[found.size()][]);
            } finally { results.release(); }
        }
    }

    // Read raw CV_32SC4 values without constructing org.opencv.core.Rect.
    // The supplied OpenCV JAR's Rect object fields can be zero even when
    // Mat.get() returns valid native rectangle coordinates.
    private static double[][] readFaceRects(MatOfRect faces) {
        java.util.ArrayList<double[]> valid = new java.util.ArrayList<double[]>();
        int rows = faces.rows();
        int cols = faces.cols();
        if (cols <= 0) return new double[0][];
        int count = Math.min(rows * cols, 10000);
        for (int i = 0; i < count; i++) {
            double[] v = faces.get(i / cols, i % cols);
            if (v == null || v.length < 4) continue;
            if (Double.isNaN(v[0]) || Double.isNaN(v[1]) ||
                Double.isNaN(v[2]) || Double.isNaN(v[3])) continue;
            if (v[0] >= 0 && v[1] >= 0 && v[2] > 0 && v[3] > 0) {
                valid.add(new double[] {v[0], v[1], v[2], v[3]});
            }
        }
        return valid.toArray(new double[valid.size()][]);
    }

    private void detectSelectedImage() {
        if (!imageMode || selectedBitmap == null) return;
        if (!opencvReady) { status.setText("OpenCV unavailable"); return; }
        final Bitmap bitmap = selectedBitmap;
        final int generation = ++imageGeneration;
        final double factor = scaleFactor;
        final int neighbors = minNeighbors, minSize = minFace;
        new Thread(new Runnable() {
            public void run() {
                Mat rgba = null, gray = null, small = null;
                MatOfRect faces = null;
                try {
                    // Convert Bitmap to RGBA Mat without org.opencv.android.Utils.
                    // Use Java 7-compatible Android Bitmap APIs and Mat.put().
                    int bw = bitmap.getWidth();
                    int bh = bitmap.getHeight();
                    int[] pixels = new int[bw * bh];
                    bitmap.getPixels(pixels, 0, bw, 0, 0, bw, bh);
                    byte[] rgbaBytes = new byte[bw * bh * 4];
                    for (int p = 0; p < pixels.length; p++) {
                        int color = pixels[p];
                        int offset = p * 4;
                        rgbaBytes[offset] = (byte) ((color >> 16) & 255);
                        rgbaBytes[offset + 1] = (byte) ((color >> 8) & 255);
                        rgbaBytes[offset + 2] = (byte) (color & 255);
                        rgbaBytes[offset + 3] = (byte) ((color >>> 24) & 255);
                    }
                    rgba = new Mat(bh, bw, CvType.CV_8UC4);
                    rgba.put(0, 0, rgbaBytes);
                    gray = new Mat();
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
                    small = new Mat();
                    double resize = Math.min(1.0, 800.0 / Math.max(gray.cols(), gray.rows()));
                    Imgproc.resize(gray, small, new Size(), resize, resize, Imgproc.INTER_LINEAR);
                    if (!useYuNet) Imgproc.equalizeHist(small, small);
                    double[][] found;
                    Mat detectorFrame = null;
                    if (useYuNet) {
                        Mat bgr = new Mat();
                        try {
                            Mat color = new Mat();
                            try {
                                Imgproc.cvtColor(rgba, color, Imgproc.COLOR_RGBA2BGR);
                                Imgproc.resize(color, bgr, new Size(small.cols(), small.rows()));
                                found = detectYuNet(bgr);
                                detectorFrame = new Mat();
                                bgr.copyTo(detectorFrame);
                            } finally { color.release(); }
                        } finally { bgr.release(); }
                    } else {
                        faces = new MatOfRect();
                        synchronized (classifier) {
                            classifier.detectMultiScale(small, faces, factor, neighbors, 0,
                                new Size(Math.max(1, minSize * resize), Math.max(1, minSize * resize)), new Size());
                        }
                        found = readFaceRects(faces);
                        detectorFrame = new Mat();
                        small.copyTo(detectorFrame);
                    }
                    // v9.7: Draw directly onto OpenCV detector Mat, BEFORE Android rendering.
                    // Green = raw rectangle, red cross = raw top-left coordinate.
                    // This isolates native detector coordinates from ImageFaceView overlays.
                    if (detectorFrame != null) {
                        Scalar green = detectorFrame.channels() == 1 ? new Scalar(255) : new Scalar(0, 255, 0);
                        Scalar cross = detectorFrame.channels() == 1 ? new Scalar(150) : new Scalar(0, 0, 255);
                        for (int j = 0; j < found.length; j++) {
                            double[] r = found[j];
                            double x = r[0], y = r[1], w = r[2], h = r[3];
                            Imgproc.rectangle(detectorFrame, new Point(x, y), new Point(x + w, y + h), green, 3);
                            if (meshEnabled && useYuNet) {
                                float[][] points = estimateMeshes(detectorFrame,
                                    new double[][] {r}, false);
                                if (points.length > 0) Rendering.drawModelMesh(detectorFrame, points[0]);
                            }
                            // Crosshair diagnostics removed.

                        }
                    }
                    // Build a bitmap from the exact pixels supplied to the detector.
                    // Avoid MatOfRect/Rect conversions and any original-photo mapping.
                    final Bitmap detectorBitmap;
                    try {
                        int dw = detectorFrame.cols(), dh = detectorFrame.rows();
                        byte[] rgb = new byte[dw * dh * detectorFrame.channels()];
                        detectorFrame.get(0, 0, rgb);
                        int[] argb = new int[dw * dh];
                        if (detectorFrame.channels() == 1) {
                            for (int k = 0; k < argb.length; k++) {
                                int v = rgb[k] & 255;
                                argb[k] = Color.rgb(v, v, v);
                            }
                        } else {
                            for (int k = 0; k < argb.length; k++) {
                                int b = rgb[k * 3] & 255;
                                int g = rgb[k * 3 + 1] & 255;
                                int r = rgb[k * 3 + 2] & 255;
                                argb[k] = Color.rgb(r, g, b);
                            }
                        }
                        detectorBitmap = Bitmap.createBitmap(argb, dw, dh, Bitmap.Config.ARGB_8888);
                    } finally { if (detectorFrame != null) detectorFrame.release(); }
                    final RectF[] boxes = new RectF[found.length];
                    final int rawFaceCount = faces == null ? found.length : (int)faces.total();
                    final int matType = faces == null ? 0 : faces.type();
                    final int matRows = faces == null ? 0 : faces.rows();
                    final int matCols = faces == null ? 0 : faces.cols();
                    final int matChannels = faces == null ? 0 : faces.channels();
                    final double[] rawFirst = faces != null && rawFaceCount > 0 ? faces.get(0, 0) : null;
                    for (int i = 0; i < found.length; i++) {
                        double[] r = found[i];
                        boxes[i] = new RectF((float)(r[0] / small.cols()), (float)(r[1] / small.rows()),
                            (float)((r[0]+r[2]) / small.cols()), (float)((r[1]+r[3]) / small.rows()));
                    }
                    final StringBuilder imageDebug = new StringBuilder();
                    imageDebug.append("Bitmap ").append(bw).append("x").append(bh)
                        .append(" | detect ").append(small.cols()).append("x").append(small.rows())
                        .append(String.format(java.util.Locale.US, " | resize %.4f", resize));
                    if (faces != null) {
                        imageDebug.append(" | Mat ").append(matRows).append("x").append(matCols)
                            .append(" ch=").append(matChannels).append(" type=").append(matType);
                    }
                    for (int i = 0; i < Math.min(found.length, 5); i++) {
                        double[] r = found[i];
                        RectF b = boxes[i];
                        imageDebug.append(String.format(java.util.Locale.US,
                            "\n#%d raw [%.0f,%.0f,%.0f,%.0f] -> img [%.0f,%.0f,%.0f,%.0f]",
                            i + 1, r[0],r[1],r[2],r[3],
                            b.left*bw,b.top*bh,b.right*bw,b.bottom*bh));
                    }
                    if (useYuNet && latestYuNetRows.length > 0) {
                        double[] r = latestYuNetRows[0];
                        imageDebug.append(String.format(java.util.Locale.US,
                            "\nLandmarks eyes (%.0f,%.0f) (%.0f,%.0f) nose (%.0f,%.0f) mouth (%.0f,%.0f) (%.0f,%.0f)",
                            r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]));
                    }
                    runOnUiThread(new Runnable() {
                        public void run() {
                            if (imageMode && generation == imageGeneration) {
                                // Draw using the SAME view that displays the photograph.
                                // Diagnostic yellow frame is deliberately independent of detections.
                                imageView.setImage(detectorBitmap);
                                imageView.setFaces(new RectF[0]); // Already drawn on native Mat.
                                imageView.setDiagnostic(false);
                                status.setText("YuNet image faces: " + boxes.length);
                            }
                        }
                    });
                } catch (final Throwable e) {
                    runOnUiThread(new Runnable() {
                        public void run() { if (generation == imageGeneration) status.setText("Image detection error: " + e.getMessage()); }
                    });
                } finally {
                    if (faces != null) faces.release();
                    if (small != null) small.release();
                    if (gray != null) gray.release();
                    if (rgba != null) rgba.release();
                }
            }
        }, "ImageFaceDetection").start();
    }

    private static class ImageFaceView extends View {
        private Bitmap bitmap;
        private RectF[] faces = new RectF[0];
        private boolean diagnostic = false;
        private Paint imagePaint = new Paint(3);
        private Paint boxPaint = new Paint(3);
        ImageFaceView(Context context) {
            super(context);
            setLayerType(View.LAYER_TYPE_SOFTWARE, null);
            boxPaint.setColor(Color.GREEN);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(6f);
            boxPaint.setAntiAlias(true);
        }
        void setImage(Bitmap value) {
            bitmap = value;
            faces = new RectF[0];
            invalidate();
        }
        void setDiagnostic(boolean enabled) { diagnostic = enabled; invalidate(); }
        void setFaces(RectF[] value) {
            faces = value == null ? new RectF[0] : value;
            invalidate();
        }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            if (bitmap == null || bitmap.isRecycled()) return;
            int vw = getWidth(), vh = getHeight();
            if (vw <= 0 || vh <= 0) return;
            float bw = bitmap.getWidth(), bh = bitmap.getHeight();
            float factor = Math.min(vw / bw, vh / bh);
            float left = (vw - bw * factor) / 2f;
            float top = (vh - bh * factor) / 2f;
            canvas.save();
            canvas.translate(left, top);
            canvas.scale(factor, factor);
            canvas.drawBitmap(bitmap, 0f, 0f, imagePaint);
            if (diagnostic) {
                // Yellow inset border: proves that the custom image view redraws.
                Paint test = new Paint(Paint.ANTI_ALIAS_FLAG);
                test.setColor(Color.YELLOW);
                test.setStyle(Paint.Style.STROKE);
                test.setStrokeWidth(Math.max(5f / factor, 3f));
                canvas.drawRect(bw * 0.06f, bh * 0.06f, bw * 0.94f, bh * 0.94f, test);
            }

            boxPaint.setStrokeWidth(Math.max(8f / factor, 4f));
            Paint idPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            idPaint.setColor(Color.YELLOW);
            idPaint.setTextSize(Math.max(32f / factor, 22f));
            idPaint.setStyle(Paint.Style.FILL);
            for (int i = 0; i < faces.length; i++) {
                RectF r = faces[i];
                float x1 = Math.max(0f, Math.min(bw, r.left * bw));
                float y1 = Math.max(0f, Math.min(bh, r.top * bh));
                float x2 = Math.max(0f, Math.min(bw, r.right * bw));
                float y2 = Math.max(0f, Math.min(bh, r.bottom * bh));
                if (x2 > x1 && y2 > y1) {
                    canvas.drawRect(x1, y1, x2, y2, boxPaint);
                    canvas.drawText("#" + (i + 1), x1 + 3f, Math.max(35f / factor, y1 - 5f), idPaint);
                }
            }
            canvas.restore();
        }
    }

    // MediaPipe-style 468 landmark inference. Output coordinates are relative to
    // a 192x192 crop; Z is model-relative depth, not metric distance.
    private void ensureFaceMeshModel() throws IOException {
        synchronized (meshLock) {
            if (faceMeshNet != null) return;
            File target = new File(getFilesDir(), "face_mesh_468.onnx");
            if (!target.exists() || target.length() < 1000000) {
                InputStream input = getAssets().open("face_mesh_Nx3x192x192.onnx");
                FileOutputStream output = new FileOutputStream(target);
                try {
                    byte[] buffer = new byte[16384]; int count;
                    while ((count = input.read(buffer)) != -1) output.write(buffer,0,count);
                } finally { try { output.close(); } finally { input.close(); } }
            }
            faceMeshNet = Dnn.readNetFromONNX(target.getAbsolutePath());
            if (faceMeshNet.empty()) throw new IOException("3D face mesh model failed to load");
        }
    }

    private float[][] estimateMeshes(Mat source, double[][] faces, boolean flipY) {
        if (!meshEnabled || faces.length == 0) { meshDiagnostic = meshEnabled ? "NO FACE" : "OFF"; return new float[0][]; }
        meshDiagnostic = "RUNNING";
        float[][] meshes = new float[Math.min(faces.length, 3)][];
        try {
            ensureFaceMeshModel();
            for (int f = 0; f < meshes.length; f++) {
                double[] r = faces[f];
                // Expand YuNet rectangle to include forehead, cheeks and jaw.
                double side = Math.max(r[2],r[3]) * 1.55;
                double cx = r[0]+r[2]*0.5, cy = r[1]+r[3]*0.5;
                double x0 = cx-side*0.5, y0 = cy-side*0.5;
                Mat patch = new Mat(192,192,CvType.CV_8UC3,new Scalar(0,0,0));
                Mat resized = new Mat(); Mat rgb = new Mat(); Mat blob = null;
                try {
                    // Clamp crop and preserve its original coordinate transform.
                    int sx=Math.max(0,(int)Math.floor(x0)), sy=Math.max(0,(int)Math.floor(y0));
                    int ex=Math.min(source.cols(),(int)Math.ceil(x0+side));
                    int ey=Math.min(source.rows(),(int)Math.ceil(y0+side));
                    if (ex<=sx || ey<=sy) continue;
                    org.opencv.core.Rect roi=new org.opencv.core.Rect(sx,sy,ex-sx,ey-sy);
                    Mat cropped=source.submat(roi);
                    try { Imgproc.resize(cropped,resized,new Size(
                        Math.max(1,Math.round((ex-sx)*192.0/side)),
                        Math.max(1,Math.round((ey-sy)*192.0/side)))); }
                    finally { cropped.release(); }
                    int dx=Math.max(0,(int)Math.round((sx-x0)*192.0/side));
                    int dy=Math.max(0,(int)Math.round((sy-y0)*192.0/side));
                    int rw=Math.min(resized.cols(),192-dx), rh=Math.min(resized.rows(),192-dy);
                    if (rw<=0 || rh<=0) continue;
                    if (resized.channels()==1) Imgproc.cvtColor(resized,resized,Imgproc.COLOR_GRAY2BGR);
                    Mat dest=patch.submat(new org.opencv.core.Rect(dx,dy,rw,rh));
                    Mat srcPart=resized.submat(new org.opencv.core.Rect(0,0,rw,rh));
                    try {srcPart.copyTo(dest);} finally {dest.release();srcPart.release();}
                    // The front-camera detector image is upside down relative to
                    // the visible preview. The mesh model must see an upright face.
                    if (flipY) org.opencv.core.Core.flip(patch, patch, 0);
                    Imgproc.cvtColor(patch,rgb,Imgproc.COLOR_BGR2RGB);
                    blob=Dnn.blobFromImage(rgb,1.0/255.0,new Size(192,192),new Scalar(0,0,0),false,false);
                    java.util.List<Mat> outs=new java.util.ArrayList<Mat>();
                    synchronized(meshLock) {
                        faceMeshNet.setInput(blob);
                        faceMeshNet.forward(outs,faceMeshNet.getUnconnectedOutLayersNames());
                    }
                    try {
                        if (outs.size()<2) { meshDiagnostic="OUTPUTS="+outs.size(); continue; }
                        Mat landmarks=outs.get(0), confidence=outs.get(1);
                        double[] score=confidence.get(0,0);
                        // The model emits a logit for face presence.
                        if (score==null || score.length==0) { meshDiagnostic="NO SCORE"; continue; }
                        // Face presence is a raw logit, not a 0..1 probability.
                        double probability = 1.0 / (1.0 + Math.exp(-score[0]));
                        if (probability < 0.10) {
                            meshDiagnostic = "FACE PROBABILITY " + String.format(java.util.Locale.US, "%.0f%%", probability * 100.0);
                            continue;
                        }
                        // The ONNX tensor is [1,468,3]. Java Mat.reshape(cn,rows)
                        // cannot reshape a 3-dimensional Mat using this overload.
                        // Read the tensor directly using multi-dimensional indices.
                        if (landmarks.dims() != 3 || landmarks.size(0) != 1
                                || landmarks.size(1) != 468 || landmarks.size(2) != 3) {
                            meshDiagnostic = "LANDMARK SHAPE " + landmarks.dims() + "D / " + landmarks.total();
                            continue;
                        }
                        float[] result = new float[468 * 3];
                        boolean valid = true;
                        for (int k = 0; k < 468; k++) {
                            double[] vx = landmarks.get(new int[] {0,k,0});
                            double[] vy = landmarks.get(new int[] {0,k,1});
                            double[] vz = landmarks.get(new int[] {0,k,2});
                            if (vx == null || vy == null || vz == null
                                    || vx.length == 0 || vy.length == 0 || vz.length == 0) {
                                valid = false; break;
                            }
                            double px = x0 + vx[0] * side / 192.0;
                            // Undo crop flip to recover detector coordinates;
                            // the final preview mapping then flips Y once.
                            double py = y0 + (flipY ? (192.0 - vy[0]) : vy[0]) * side / 192.0;
                            if (Double.isNaN(px) || Double.isNaN(py)
                                    || Double.isInfinite(px) || Double.isInfinite(py)) {
                                valid = false; break;
                            }
                            result[3*k] = (float)(px / source.cols());
                            result[3*k+1] = (float)(flipY ? 1.0 - py / source.rows() : py / source.rows());
                            result[3*k+2] = (float)(vz[0] * side / 192.0 / source.cols());
                        }
                        if (!valid) result = null;
                        meshes[f]=result;
                        meshDiagnostic=result==null ? "INVALID LANDMARKS" : "468 LANDMARKS / "+(Rendering.MESH_EDGES.length/2)+" EDGES / "+String.format(java.util.Locale.US,"%.0f%%",probability*100.0);
                    } finally {for(Mat m:outs)m.release();}
                } finally {if(blob!=null)blob.release();rgb.release();resized.release();patch.release();}
            }
        } catch(Throwable e) {
            meshDiagnostic="ERROR: "+e.getClass().getSimpleName()+": "+e.getMessage();
        }
        return meshes;
    }


}